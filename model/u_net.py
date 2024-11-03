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
        x4 = self.encoder.down4(x3)  
        x5 = self.encoder.bottleneck(x4)  

        # Decoder path
        x = self.decoder.up1(x5)
        x = torch.cat([x4, x], dim=1)
        x = self.decoder.conv1(x)

        x = self.decoder.up2(x)
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

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # (N, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # (N, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # (N, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.down4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # (N, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # (N, 512, 2, 2)
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # (N, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1),  # (N, 512, 4, 4)
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # (N, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1),  # (N, 256, 8, 8)
            nn.BatchNorm2d(256),
            nn.ReLU()
        )
        self.up3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # (N, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1),  # (N, 128, 16, 16)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up4 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # (N, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1),  # (N, 64, 32, 32)
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.final_up = nn.Sequential(
            nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # (N, 3, 64, 64)
            nn.Tanh()
        )

if __name__ == "__main__":
    model = UNet()
    print(model)

    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(y.shape)  # Should output torch.Size([1, 3, 64, 64])
