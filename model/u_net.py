import torch
from torch import nn

"""
    U-Net architecture that is kept close to the parameters of the DCGAN.
    Note that usual ReLUs are used here.
    TODO: Check if strides are counterproductive for U-Net
"""
class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.weight_init()

    def forward(self, x):
        # Encoder path
        x1 = self.encoder.down1(x) 
        x2 = self.encoder.down2(x1)  
        x3 = self.encoder.down3(x2)  
        x4 = self.encoder.bottleneck(x3)  

        # Decoder path
        x = self.decoder.up2(x4)
        x = torch.cat([x3, x], dim=1)
        x = self.decoder.conv2(x)

        x = self.decoder.up3(x)
        x = torch.cat([x2, x], dim=1)
        x = self.decoder.conv3(x)

        x = self.decoder.up4(x)
        x = torch.cat([x1, x], dim=1)
        x = self.decoder.conv4(x)

        x = self.decoder.final_up(x)
        return x

    def weight_init(self):
        # Initialize weights as in DCGAN
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(DownBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding),
            #nn.BatchNorm2d(out_channels),
            nn.GroupNorm(1, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.seq(x)
        return x
        

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down1 = DownBlock(3, 64, kernel_size=4, stride=2, padding=1)  # (N, 64, 32, 32)

        self.down2 = DownBlock(64, 128, kernel_size=4, stride=2, padding=1)  # (N, 128, 16, 16)
 
        self.down3 = DownBlock(128, 256, kernel_size=4, stride=2, padding=1)  # (N, 256, 8, 8)

        #self.down4 = DownBlock(256, 512, kernel_size=4, stride=2, padding=1)  # (N, 512, 4, 4)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1),  # (N, 512, 2, 2)
            #nn.BatchNorm2d(512),
            nn.GroupNorm(1, 256),
            nn.ReLU()
        )

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding):
        super(ConvBlock, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),  # (N, 512, 4, 4)
            #nn.BatchNorm2d(512),
            nn.GroupNorm(1, out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.seq(x)
        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        # self.up1 = nn.Sequential(
        #     nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # (N, 512, 4, 4)
        #     #nn.BatchNorm2d(512),
        #     nn.GroupNorm(1, 512),
        #     nn.ReLU()
        # )
        # self.conv1 = ConvBlock(512 + 512, 512, kernel_size=3, padding=1)  # (N, 512, 4, 4)

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1),  # (N, 256, 8, 8)
            #nn.BatchNorm2d(256),
            nn.GroupNorm(1, 256),
            nn.ReLU()
        )
        self.conv2 = ConvBlock(256 + 256, 256, kernel_size=3, padding=1)  # (N, 256, 8, 8)
   
   
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (N, 128, 16, 16)
            #nn.BatchNorm2d(128),
            nn.GroupNorm(1, 128),
            nn.ReLU()
        )
        self.conv3 = ConvBlock(128 + 128, 128, kernel_size=3, padding=1)  # (N, 128, 16, 16)

        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (N, 64, 32, 32)
            #nn.BatchNorm2d(64),
            nn.GroupNorm(1, 64),
            nn.ReLU()
        )
        self.conv4 = ConvBlock(64 + 64, 64, kernel_size=3, padding=1)  # (N, 64, 32, 32)

        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # (N, 3, 64, 64)
            nn.Tanh()
        )

if __name__ == "__main__":
    model = UNet()
    print(model)

    x = torch.randn(5, 3, 64, 64)
    y = model(x)
    print(y.shape)  # Should output torch.Size([5, 3, 64, 64])
