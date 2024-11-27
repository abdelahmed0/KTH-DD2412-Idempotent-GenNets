import torch
from torch import nn

from model.modules import ConvBlock

"""
    DCGAN implementation based on the 
    "Unsupervised representation learning with Deep Convolutional Generative Adversarial Networks" paper

    Parameters for the DCGAN model are taken from the "Idempotent Generative Networks" paper
    MNIST Layout stolen from https://github.com/kpandey008/dcgan 
    and then modified by including Dropout and GroupNorm layers
"""
class DCGAN(nn.Module):
    def __init__(self, architecture='DCGAN', input_size=64, norm='batchnorm', use_bias=True):
        super(DCGAN, self).__init__()
        if architecture == 'DCGAN_MNIST' or architecture == 'DCGAN_MNIST_2':
            num_channels = 1
        elif architecture == 'DCGAN':
            num_channels = 3
        else:
            raise NotImplementedError(f"Architecture {architecture} is not supported yet.")

        self.encoder = Encoder(num_channels, input_size, architecture, norm, use_bias)
        self.decoder = Decoder(num_channels, input_size, architecture, norm, use_bias)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Encoder(nn.Module):
    """ 
        Discriminator for DCGAN
        Uses strided convolutions instead of pooling layers and no fully connected layers
    """
    def __init__(self, num_channels, input_size, architecture, norm, use_bias):
        super(Encoder, self).__init__()
        layers = []

        if architecture == 'DCGAN_MNIST':
            layers += [
                nn.Conv2d(num_channels, input_size, 4, 2, 1, bias=use_bias),  # [1,28,28] -> [64,14,14]
                nn.Dropout2d(0.1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.GroupNorm(num_groups=8, num_channels=input_size),
                nn.Conv2d(input_size, input_size * 2, 4, 2, 1, bias=use_bias),  # [64,14,14] -> [128,7,7]
                nn.Dropout2d(0.1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.GroupNorm(num_groups=8, num_channels=input_size * 2),
                nn.Conv2d(input_size * 2, input_size * 4, 3, 1, 0, bias=use_bias),  # [128,7,7] -> [256,5,5]
                nn.Dropout2d(0.1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.GroupNorm(num_groups=8, num_channels=input_size * 4),
                nn.Conv2d(input_size * 4, input_size * 8, 3, 1, 0, bias=use_bias),  # [256,5,5] -> [512,3,3]
                nn.Dropout2d(0.1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),

                nn.Conv2d(input_size * 8, input_size * 8, 3, 1, 0, bias=use_bias), # [512,3,3] -> [512,1,1]
            ]
        elif architecture == 'DCGAN':
            # Encoder for DCGAN (3x64x64)
            layers += [
                ConvBlock(num_channels, input_size, 4, 2, 1, norm='none', activation='leakyrelu', bias=use_bias),
                ConvBlock(input_size, input_size*2, 4, 2, 1, norm=norm, activation='leakyrelu', bias=use_bias),
                ConvBlock(input_size*2, input_size*4, 4, 2, 1, norm=norm, activation='leakyrelu', bias=use_bias),
                ConvBlock(input_size*4, input_size*8, 4, 2, 1, norm=norm, activation='leakyrelu', bias=use_bias),
                ConvBlock(input_size*8, input_size*8, 4, 1, 0, norm='none', activation='none', bias=use_bias),
            ]
        elif architecture == 'DCGAN_MNIST_2':
            # Encoder for MNIST (1x28x28)
            layers += [
                ConvBlock(num_channels, input_size, 4, 2, 1, norm="none", activation='leakyrelu', bias=use_bias),
                ConvBlock(input_size, input_size*2, 4, 2, 1, norm=norm, activation='leakyrelu', bias=use_bias),
                ConvBlock(input_size*2, input_size*4, 3, 2, 1, norm=norm, activation='leakyrelu', bias=use_bias),
                ConvBlock(input_size*4, input_size*4, 4, 1, 0, norm="none", activation='none', bias=use_bias),
            ]

            # layers += [
            #     ConvBlock(num_channels, input_size, 4, 2, 1, norm='none', activation='leakyrelu', bias=use_bias),
            #     ConvBlock(input_size, input_size*2, 4, 2, 1, norm=norm, activation='leakyrelu', bias=use_bias),
            #     ConvBlock(input_size*2, input_size*4, 3, 1, 0, norm=norm, activation='leakyrelu', bias=use_bias),
            #     ConvBlock(input_size*4, input_size*8, 3, 1, 0, norm=norm, activation='leakyrelu', bias=use_bias),
            #     ConvBlock(input_size*8, input_size*8, 3, 1, 0, norm='none', activation='none', bias=use_bias),
            # ]

        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """ 
        Generator for DCGAN
        Uses fractional-strided convolutions instead of pooling layers and no fully connected layers
    """
    def __init__(self, num_channels, input_size, architecture, norm, use_bias):
        super(Decoder, self).__init__()
        layers = []
        
        if architecture == 'DCGAN_MNIST':
            in_channels = input_size * 8
            layers += [
                nn.GroupNorm(num_groups=8, num_channels=in_channels),
                nn.ConvTranspose2d(in_channels, in_channels // 2, 3, 1, 0, bias=use_bias),  # [512,1,1] -> [256,3,3]
                nn.Dropout2d(0.1),
                nn.ReLU(True),

                nn.GroupNorm(num_groups=8, num_channels=in_channels // 2),
                nn.ConvTranspose2d(in_channels // 2, in_channels // 4, 3, 1, 0, bias=use_bias),  # [256,3,3] -> [128,5,5]
                nn.Dropout2d(0.1),
                nn.ReLU(True),

                nn.GroupNorm(num_groups=8, num_channels=in_channels // 4),
                nn.ConvTranspose2d(in_channels // 4, in_channels // 8, 3, 1, 0, bias=use_bias),  # [128,5,5] -> [64,7,7]
                nn.Dropout2d(0.1),
                nn.ReLU(True),

                nn.GroupNorm(num_groups=8, num_channels=in_channels // 8),
                nn.ConvTranspose2d(in_channels // 8, in_channels // 16, 4, 2, 1, bias=use_bias),  # [64,7,7] -> [32,14,14]
                nn.Dropout2d(0.1),
                nn.ReLU(True),

                nn.ConvTranspose2d(in_channels // 16, num_channels, 4, 2, 1, bias=use_bias),  # [32,14,14] -> [1,28,28]
                nn.Tanh(),
            ]
        elif architecture == 'DCGAN':
            # Decoder for DCGAN (3x64x64)
            layers += [
                ConvBlock(input_size*8, input_size*8, 4, 1, 0, norm=norm, activation='relu', transposed=True, bias=use_bias),
                ConvBlock(input_size*8, input_size*4, 4, 2, 1, norm=norm, activation='relu', transposed=True, bias=use_bias),
                ConvBlock(input_size*4, input_size*2, 4, 2, 1, norm=norm, activation='relu', transposed=True, bias=use_bias),
                ConvBlock(input_size*2, input_size, 4, 2, 1, norm=norm, activation='relu', transposed=True, bias=use_bias),
                ConvBlock(input_size, num_channels, 4, 2, 1, norm='none', activation='tanh', transposed=True, bias=use_bias),
            ]
        elif architecture == 'DCGAN_MNIST_2':
            # Decoder for MNIST (1x28x28)
            layers += [
                ConvBlock(input_size*4, input_size*4, 4, 1, 0, norm=norm, activation='relu', transposed=True, bias=use_bias),
                ConvBlock(input_size*4, input_size*2, 3, 2, 1, norm=norm, activation='relu', transposed=True, bias=use_bias),
                ConvBlock(input_size*2, input_size, 4, 2, 1, norm=norm, activation='relu', transposed=True, bias=use_bias),
                ConvBlock(input_size, num_channels, 4, 2, 1, norm="none", activation='tanh', transposed=True, bias=use_bias),
            ]

            # in_channels = input_size * 8
            # layers += [
            #     ConvBlock(in_channels, in_channels // 2, 3, 1, 0, norm=norm, activation='relu', transposed=True, bias=use_bias),
            #     ConvBlock(in_channels // 2, in_channels // 4, 3, 1, 0, norm=norm, activation='relu', transposed=True, bias=use_bias),
            #     ConvBlock(in_channels // 4, in_channels // 8, 3, 1, 0, norm=norm, activation='relu', transposed=True, bias=use_bias),
            #     ConvBlock(in_channels // 8, in_channels // 16, 4, 2, 1, norm=norm, activation='relu', transposed=True, bias=use_bias),
            #     ConvBlock(in_channels // 16, num_channels, 4, 2, 1, norm='none', activation='tanh', transposed=True, bias=use_bias),
            # ]
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    from torchsummary import summary

    model = DCGAN()
    x = torch.randn(5, 3, 64, 64)
    y = model(x)
    summary(model, (3, 64, 64), device='cpu')
    print("DCGAN", y.shape)
    del model

    model_mnist = DCGAN(architecture='DCGAN_MNIST')
    x = torch.randn(5, 1, 28, 28)
    y = model_mnist(x)
    print("DCGAN_MNIST", y.shape)
    del model_mnist

    model_mnist_2 = DCGAN(architecture='DCGAN_MNIST_2')
    x = torch.randn(5, 1, 28, 28)
    y = model_mnist_2(x)
    print("DCGAN_MNIST_2", y.shape)
    del model_mnist_2