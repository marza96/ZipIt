from torch import nn


class MLP(nn.Module):
    def __init__(self, channels=128, layers=3, classes=10, bnorm=False):
        super().__init__()
        
        self.bnorm      = bnorm
        self.classes    = classes
        self.channels   = channels
        self.num_layers = layers
        
        mid_layers = [
            nn.Linear(28 * 28, channels, bias=True),
        ]
        if self.bnorm is True:
            mid_layers.append(nn.BatchNorm1d(channels))

        mid_layers.append(nn.ReLU())

        for i in range(layers):
            lst  = [
                nn.Linear(channels, channels, bias=True),
            ]
            if self.bnorm is True:
                lst.append(nn.BatchNorm1d(channels))
                
            lst.append(nn.ReLU())

            if i == self.num_layers - 1:
                lst = [
                    nn.Linear(channels, channels, bias=True),
                ]
                if self.bnorm is True:
                    lst.append(nn.BatchNorm1d(channels))

            mid_layers.extend(lst)
            
        self.layers = nn.Sequential(*mid_layers)

    def forward(self, x):
        if x.size(1) == 3:
            x = x.mean(1, keepdim=True)

        x = x.reshape(x.size(0), -1)

        x = self.layers(x)
 
        return x