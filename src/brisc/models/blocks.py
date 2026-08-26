import torch
import torch.nn as nn

class CNNBlock(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(output_channels),
            nn.ReLU(inplace=True),
            # nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class CNNEncoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.encoder = CNNBlock(in_ch, out_ch)
        self.down = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        encode = self.encoder(x)
        down = self.down(encode)
        return encode, down

class CNNDecoder(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=2, stride=2)
        self.decode = CNNBlock(out_ch + skip_ch, out_ch)

    def forward(self, x, encode):
        up = self.up(x)
        if x.shape[-2:] != encode.shape[-2:]:
            encode = nn.functional.pad(encode, (0, up.shape[3] - encode.shape[3], 0, up.shape[2] - encode.shape[2]),)
        x = torch.cat([up, encode], dim=1)
        x = self.decode(x)
        return x