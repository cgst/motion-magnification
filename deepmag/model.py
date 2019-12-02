import torch
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


class Manipulator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, 3, 1),
            nn.ReLU(),
        )
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(32, 32, 3, 1),
            ResidualBlock(32, 32, 3, 1),
        )

    def forward(self, shape_a, shape_b, amp_f):
        g = self.conv1(shape_b - shape_a)
        amp = amp_f.reshape(len(amp_f), 1, 1, 1)
        return shape_a + self.conv2(g * amp)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2)
        self.res = nn.Sequential(
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
            ResidualBlock(64, 64, 3, 1),
        )
        self.deconv = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64, 32, 3, 1),
            nn.ReLU(),
            nn.ReflectionPad2d(3),
            nn.Conv2d(32, 3, 7, 1),
        )

    def forward(self, texture, shape):
        y = self.upsample(texture)
        y = torch.cat((y, shape), dim=1)
        y = self.res(y)
        y = self.deconv(y)
        return y


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


class MagNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.manipulator = Manipulator()
        self.decoder = Decoder()

    def forward(self, frame_a, frame_b, amp_f):
        texture_a, shape_a = self.encoder(frame_a)
        texture_b, shape_b = self.encoder(frame_b)
        shape_amp = self.manipulator(shape_a, shape_b, amp_f)
        frame_amp = self.decoder(texture_b, shape_amp)
        return frame_amp, (texture_a, shape_a), (texture_b, shape_b)
