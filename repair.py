import tqdm
import torch
import torch.nn as nn


def reset_bn_stats(model, loader, epochs=1, device="cpu"):
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d or type(m) == nn.BatchNorm1d:
            m.momentum = None # use simple average
            m.reset_running_stats()

    # run a single train epoch with augmentations to recalc stats
    model.train()
    for _ in range(epochs):
        with torch.no_grad():
            for images, _ in tqdm.tqdm(loader):
                output = model(images.to(device))
    

def make_tracked_net(net, model_cls, layer_indices, device="cpu"):
    net1 = model_cls(layers=len(layer_indices) - 1, classes=net.classes).to(device)
    net1.load_state_dict(net.state_dict())
    for i in range(len(layer_indices)):
        idx = layer_indices[i]
        if isinstance(net1.layers[idx], nn.Linear):
            net1.layers[idx] = ResetLinear(net1.layers[idx])
    return net1.to(device).eval()


class ResetLinear(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.h = h = layer.out_features if hasattr(layer, 'out_features') else 768
        self.layer = layer
        self.bn = nn.BatchNorm1d(h)
        self.rescale = False
        
    def set_stats(self, goal_mean, goal_var):
        self.bn.bias.data = goal_mean
        goal_std = (goal_var + 1e-5).sqrt()
        self.bn.weight.data = goal_std
        
    def forward(self, *args, **kwargs):
        x = self.layer(*args, **kwargs)
        if self.rescale:
            x = self.bn(x)
        else:
            self.bn(x)
        return x
    

def repair(model0, model1, model_a, model_cls, layer_indices, loader0, loader1, loaderc, device="cpu"):
    ## calculate the statistics of every hidden unit in the endpoint networks
    ## this is done practically using PyTorch BatchNorm2d layers.
    wrap0 = make_tracked_net(model0, model_cls, layer_indices, device=device)
    wrap1 = make_tracked_net(model1, model_cls, layer_indices, device=device)
    reset_bn_stats(wrap0, loader0, device=device)
    reset_bn_stats(wrap1, loader1, device=device)

    wrap_a = make_tracked_net(model_a, model_cls, layer_indices, device=device)
    ## set the goal mean/std in added bns of interpolated network, and turn batch renormalization on
    for m0, m_a, m1 in zip(wrap0.modules(), wrap_a.modules(), wrap1.modules()):
        if not isinstance(m0, ResetLinear):
            continue
        # get goal statistics -- interpolate the mean and std of parent networks
        mu0 = m0.bn.running_mean
        mu1 = m1.bn.running_mean
        goal_mean = (mu0 + mu1)/2
        var0 = m0.bn.running_var
        var1 = m1.bn.running_var
        goal_var = ((var0.sqrt() + var1.sqrt())/2).square()
        # set these in the interpolated bn controller
        m_a.set_stats(goal_mean, goal_var)
        # turn rescaling on
        m_a.rescale = True
        
    # reset the tracked mean/var and fuse rescalings back into conv layers 
    reset_bn_stats(wrap_a, loaderc, device=device)

    return wrap_a


def repair_bnorm(model0, model1, model_a, model_cls, layer_indices, loader0, loader1, loaderc, device="cpu"):
    reset_bn_stats(model_a, loaderc, device=device)

    return model_a