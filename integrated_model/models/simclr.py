import torch.nn as nn

#===============================================================================
class classification_layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(classification_layer, self).__init__()

        self.classification_layer = nn.Sequential(
                nn.Linear(in_channel, in_channel),
                nn.LeakyReLU(),
                nn.Linear(in_channel, out_channel)
            )

    def forward(self,x):
        x = self.classification_layer(x)
        return x

#===============================================================================
class fc_layer(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(fc_layer, self).__init__()

        # self.fc_layer_ = nn.Sequential(
        #        nn.Linear(in_channel, in_channel),
        #        nn.LeakyReLU())

        self.projection_layer = nn.Sequential(
                nn.Linear(in_channel, in_channel),
                nn.LeakyReLU(),
                nn.Linear(in_channel, 64)
            )

        self.classification_layer = nn.Sequential(
            nn.Linear(in_channel, in_channel),
            nn.LeakyReLU(),
            nn.Linear(in_channel, out_channel),
                # nn.Linear(in_channel, 2048),
                # nn.LeakyReLU(),
                # nn.Linear(2048, 4096),
                # nn.LeakyReLU(),
                # nn.Linear(4096, 2048),
                # nn.LeakyReLU(),
                # nn.Linear(2048, 256),
                # nn.LeakyReLU(),
                # nn.Linear(256, out_channel),
                # nn.LeakyReLU(0.4),
            )

    def forward(self,x, inference=False):
        # x = self.fc_layer_(x)
        output = self.classification_layer(x)
        projection = None
        if not inference:
            if self.projection_layer is not None:
                projection = self.projection_layer(x)
        return output, projection
