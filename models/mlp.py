from torch import nn

import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, h=128, layers=3):
        super().__init__()

        self.subnet     = Subnet

        self.h          = h
        self.num_layers = layers
        self.fc1        = nn.Linear(28*28, h, bias=True)

        mid_layers = []
        for _ in range(layers):
            mid_layers.extend([
                nn.Linear(h, h, bias=True),
                nn.ReLU(),
            ])
            
        self.layers = nn.Sequential(*mid_layers)
        self.fc2 = nn.Linear(h, 10)

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.layers(x)
        x = self.fc2(x)

        return x
    

class Subnet(nn.Module):
    def __init__(self, model, layer_i):
        super().__init__()
        self.model = model
        self.layer_i = layer_i

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = x.reshape(x.size(0), -1)
        x = F.relu(self.model.fc1(x))
        x = self.model.layers[:2 * self.layer_i](x)
        
        return x