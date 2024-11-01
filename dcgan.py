import torch
from torch import nn



""" 
    DCGAN implementation based on the 
    "Unsupervised representation learning with Deep Convolutional Generative Adversarial Networks" paper

    Parameters for the DCGAN model are taken from the "Idempotent Generative Networks" paper 
"""
class DCGAN(nn.Module):
    def __init__(self, architecture='DCGAN'):
        super(DCGAN, self).__init__()
        self.encoder = Encoder(architecture=architecture)
        self.decoder = Decoder(architecture=architecture)
        self.weight_init()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
    def weight_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0.0, 0.02)
                m.bias.data.fill_(0)


class Encoder(nn.Module):
    """ 
        Discriminator for DCGAN
        Uses strided convolutions instead of pooling layers and no fully connected layers
    """
    def __init__(self, architecture):
        super(Encoder, self).__init__()
        if 'DCGAN' == architecture:
            self.model = nn.Sequential(
                nn.Conv2d(3, 64, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.LeakyReLU(0.2),
                nn.Conv2d(256, 512, 4, 2, 1),
                nn.BatchNorm2d(512),
                nn.LeakyReLU(0.2),
                nn.Conv2d(512, 512, 4, 1, 0)
            )
        elif 'DCGAN_MNIST' == architecture:
            self.model = nn.Sequential(
                nn.Conv2d(1, 32, 4, 2, 1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(32, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.LeakyReLU(0.2),
                nn.Conv2d(64, 64, 4, 1, 0)
            )
        else:
            NotImplementedError(f"Architecture {architecture} is not supported yet.")
    
    def forward(self, x):
        return self.model(x)


class Decoder(nn.Module):
    """ 
        Generator for DCGAN
        Uses fractional-strided convolutions instead of pooling layers and no fully connected layers
    """
    def __init__(self, architecture):
        super(Decoder, self).__init__()
        if 'DCGAN' == architecture:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(512, 512, 4, 1, 0),
                nn.BatchNorm2d(512),
                nn.ReLU(),
                nn.ConvTranspose2d(512, 256, 4, 2, 1),
                nn.BatchNorm2d(256),
                nn.ReLU(),
                nn.ConvTranspose2d(256, 128, 4, 2, 1),
                nn.BatchNorm2d(128),
                nn.ReLU(),
                nn.ConvTranspose2d(128, 64, 4, 2, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 3, 4, 2, 1),
                nn.Tanh()
            )
        elif 'DCGAN_MNIST' == architecture:
            self.model = nn.Sequential(
                nn.ConvTranspose2d(64, 64, 4, 1, 0),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.ConvTranspose2d(32, 1, 4, 2, 1),
                nn.Tanh()
            )
        else:
            NotImplementedError(f"Architecture {architecture} is not supported yet.")
    
    def forward(self, x):
        return self.model(x)

if __name__ == "__main__":
    model = DCGAN()
    print(model)

    x = torch.randn(1, 3, 64, 64)
    y = model(x)
    print(y.shape)