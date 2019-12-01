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

    def forward(self, enc_shape_a, enc_shape_b, amp_f):
        g = self.conv1(enc_shape_b - enc_shape_a)
        amp = amp_f.reshape(len(amp_f), 1, 1, 1)
        return enc_shape_a + self.conv2(g * amp)


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

    def forward(self, enc_texture, enc_shape):
        y = self.upsample(enc_texture)
        y = torch.cat((y, enc_shape), dim=1)
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
