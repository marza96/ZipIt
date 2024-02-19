from torch import nn

class VGG(nn.Module):
    def __init__(self, cfg, w=1, classes=10, in_channels=3, bnorm=False):
        super().__init__()

        self.in_channels = in_channels
        self.w           = w
        self.bnorm       = bnorm
        self.classes     = classes
        self.layers      = self._make_layers(cfg)

    def forward(self, x):
        out = self.layers[:-2](x)
        out = out.view(out.size(0), -1)
        out = self.layers[-2:](out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels

        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers.append(nn.Conv2d(in_channels if in_channels == 3 else self.w*in_channels,
                                     self.w*x, kernel_size=3, padding=1))
                
                if self.bnorm is True:
                    layers.append(nn.BatchNorm2d(self.w*x))

                layers.append(nn.ReLU(inplace=True))
                in_channels = x

        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        layers += [nn.Linear(self.w * cfg[-2], self.classes)]

        if self.bnorm is True:
            layers.append(nn.BatchNorm1d(self.classes))

        return nn.Sequential(*layers)
