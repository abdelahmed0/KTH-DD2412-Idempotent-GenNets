import torch
from torch import nn



"""
    DCGAN implementation based on the 
    "Unsupervised representation learning with Deep Convolutional Generative Adversarial Networks" paper

    Parameters for the DCGAN model are taken from the "Idempotent Generative Networks" paper
    MNIST Layout stolen from https://github.com/kpandey008/dcgan 
    and then modified by including Dropout and GroupNorm layers
"""
class DCGAN(nn.Module):
    def __init__(self, architecture='DCGAN', input_size=64, use_bias=False):
        super(DCGAN, self).__init__()
        if architecture == 'DCGAN_MNIST':
            num_channels = 1
        elif architecture == 'DCGAN':
            num_channels = 3
        else:
            raise NotImplementedError(f"Architecture {architecture} is not supported yet.")
        self.encoder = Encoder(num_channels, input_size, architecture, use_bias)
        self.decoder = Decoder(num_channels, input_size, architecture, use_bias)
        self.weight_init()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()
                

class Encoder(nn.Module):
    """ 
        Discriminator for DCGAN
        Uses strided convolutions instead of pooling layers and no fully connected layers
    """
    def __init__(self, num_channels, input_size, architecture, use_bias):
        super(Encoder, self).__init__()
        layers = []

        if architecture == 'DCGAN_MNIST':
            layers += [
                nn.Conv2d(num_channels, input_size, 4, 2, 1, bias=False),  # [1,28,28] -> [64,14,14]
                nn.Dropout2d(0.1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.GroupNorm(num_groups=8, num_channels=input_size),
                nn.Conv2d(input_size, input_size * 2, 4, 2, 1, bias=False),  # [64,14,14] -> [128,7,7]
                nn.Dropout2d(0.1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.GroupNorm(num_groups=8, num_channels=input_size * 2),
                nn.Conv2d(input_size * 2, input_size * 4, 3, 1, 0, bias=False),  # [128,7,7] -> [256,5,5]
                nn.Dropout2d(0.1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.GroupNorm(num_groups=8, num_channels=input_size * 4),
                nn.Conv2d(input_size * 4, input_size * 8, 3, 1, 0, bias=False),  # [256,5,5] -> [512,3,3]
                nn.Dropout2d(0.1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.Conv2d(input_size * 8, input_size * 8, 3, 1, 0, bias=False), # [512,3,3] -> [512,1,1]
            ]
        elif architecture == 'DCGAN':
            # Encoder for DCGAN (3x64x64)
            layers += [
                nn.Conv2d(num_channels, input_size, kernel_size=4, stride=2, padding=1, bias=use_bias),  # [3,64,64] -> [64,32,32]
                nn.LeakyReLU(0.2, inplace=True),

                nn.Conv2d(input_size, input_size * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),  # [64,32,32] -> [128,16,16]
                nn.LeakyReLU(0.2, inplace=True),

                nn.BatchNorm2d(input_size * 2),
                nn.Conv2d(input_size * 2, input_size * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),  # [128,16,16] -> [256,8,8]
                nn.LeakyReLU(0.2, inplace=True),

                nn.BatchNorm2d(input_size * 4),
                nn.Conv2d(input_size * 4, input_size * 8, kernel_size=4, stride=2, padding=1, bias=use_bias),  # [256,8,8] -> [512,4,4]
                nn.LeakyReLU(0.2, inplace=True),

                nn.BatchNorm2d(input_size * 8),
                nn.Conv2d(input_size * 8, input_size * 8, kernel_size=4, stride=1, padding=0, bias=use_bias)   # [512,4,4] -> [512,1,1]
            ]

        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """ 
        Generator for DCGAN
        Uses fractional-strided convolutions instead of pooling layers and no fully connected layers
    """
    def __init__(self, num_channels, input_size, architecture, use_bias):
        super(Decoder, self).__init__()
        layers = []
        
        if architecture == 'DCGAN_MNIST':
            in_channels = input_size * 8
            layers += [
                nn.GroupNorm(num_groups=8, num_channels=in_channels),
                nn.ConvTranspose2d(in_channels, in_channels // 2, 3, 1, 0, bias=False),  # [512,1,1] -> [256,3,3]
                nn.Dropout2d(0.1),
                nn.ReLU(True),

                nn.GroupNorm(num_groups=8, num_channels=in_channels // 2),
                nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 3, 1, 0, bias=False),  # [256,3,3] -> [128,5,5]
                nn.Dropout2d(0.1),
                nn.ReLU(True),

                nn.GroupNorm(num_groups=8, num_channels=in_channels // 4),
                nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 3, 1, 0, bias=False),  # [128,5,5] -> [64,7,7]
                nn.Dropout2d(0.1),
                nn.ReLU(True),

                nn.GroupNorm(num_groups=8, num_channels=in_channels // 8),
                nn.ConvTranspose2d(in_channels // 8, in_channels // 16, 4, 2, 1, bias=False),  # [64,7,7] -> [32,14,14]
                nn.Dropout2d(0.1),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels // 16, num_channels, 4, 2, 1, bias=False),  # [32,14,14] -> [1,28,28]
                nn.Tanh(),
            ]
        elif architecture == 'DCGAN':
            # Decoder for DCGAN (3x64x64)
            layers += [
                nn.ConvTranspose2d(input_size * 8, input_size * 8, kernel_size=4, stride=1, padding=0, bias=use_bias),  # [512,1,1] -> [512,4,4]
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(input_size * 8),

                nn.ConvTranspose2d(input_size * 8, input_size * 4, kernel_size=4, stride=2, padding=1, bias=use_bias),  # [512,4,4] -> [256,8,8]
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(input_size * 4),

                nn.ConvTranspose2d(input_size * 4, input_size * 2, kernel_size=4, stride=2, padding=1, bias=use_bias),  # [256,8,8] -> [128,16,16]
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(input_size * 2),

                nn.ConvTranspose2d(input_size * 2, input_size, kernel_size=4, stride=2, padding=1, bias=use_bias),      # [128,16,16] -> [64,32,32]
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(input_size),

                nn.ConvTranspose2d(input_size, num_channels, kernel_size=4, stride=2, padding=1, bias=use_bias),      # [64,32,32] -> [3,64,64]
                nn.Tanh()
            ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = DCGAN()
    print(model)

    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(y.shape)

    model_mnist = DCGAN(architecture='DCGAN_MNIST')
    x = torch.randn(1, 1, 28, 28)
    y = model_mnist(x)
    print(y.shape)