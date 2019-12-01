from torch import nn
from torch.functional import F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # Convolution 1
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 16, 7, 1),
            nn.ReLU(),
            # Convolution 2
            nn.ReflectionPad2d(1),
            nn.Conv2d(16, 32, 3, 2),
            nn.ReLU(),
        )
        self.res = nn.Sequential(
            ResidualBlock(32, 32, 3, 1),
            ResidualBlock(32, 32, 3, 1),
            ResidualBlock(32, 32, 3, 1)
        )
        self.texture = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, 3, 2),
            nn.ReLU(),
            ResidualBlock(32, 32, 3, 1),
            ResidualBlock(32, 32, 3, 1),
        )
        self.shape = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
            ResidualBlock(32, 32, 3, 1),
            ResidualBlock(32, 32, 3, 1),
        )

    def forward(self, x):
        y = self.conv(x)
        y = self.res(y)
        texture_y = self.texture(y)
        shape_y = self.shape(y)
        return texture_y, shape_y


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
            nn.ReLU(),
            nn.ReflectionPad2d(kernel_size // 2),
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride),
        )

    def forward(self, x):
        y = self.block(x)
        return x + y
