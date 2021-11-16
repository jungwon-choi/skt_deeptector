import torch.nn as nn
import torch.nn.functional as F


class OUDefend(nn.Module):
    def __init__(self, dim_input=3):
        super(OUDefend, self).__init__()
        self.conv1_u = nn.Conv2d(dim_input, 16, 3, padding =1)
        self.conv2_u = nn.Conv2d(16, 4, 3, padding =1)
        self.pool = nn.MaxPool2d(2, 2)
        self.deconv1_u = nn.ConvTranspose2d(4, 16, 2, stride=2)
        self.deconv2_u = nn.ConvTranspose2d(16, dim_input, 2, stride=2)

        self.conv1_o = nn.Conv2d(4, 16, 3, padding =1)
        self.conv2_o = nn.Conv2d(16, dim_input, 3, padding =1)
        self.deconv1_o = nn.ConvTranspose2d(dim_input, 16, 2, stride=2)
        self.deconv2_o = nn.ConvTranspose2d(16, 4, 2, stride=2)

        self.sigmoid = nn.Sigmoid()

    def Opart(self, x):
        x = F.relu(self.deconv1_o(x))
        x = F.relu(self.deconv2_o(x))
        x = F.relu(self.conv1_o(x))
        x = self.pool(x)
        x = self.conv2_o(x)
        x = self.pool(x)
        x = self.sigmoid(x)
        return x

    def Upart(self, x):
        x = F.relu(self.conv1_u(x))
        x = self.pool(x)
        x = F.relu(self.conv2_u(x))
        x = self.pool(x)
        x = F.relu(self.deconv1_u(x))
        x = self.sigmoid(self.deconv2_u(x))
        return x

    def forward(self, x):
        x_from_O = self.Opart(x)
        x_from_U = self.Upart(x)

        if x_from_O.shape != x_from_U.shape:
            x_from_U = F.interpolate(x_from_U, size=x_from_O.shape[2:])

        out = 0.5 * x_from_O + 0.5 * x_from_U
        return out
