from torch import nn

def get_activation(activation):
    if activation == 'none':
        return None
    elif activation == 'relu':
        return nn.ReLU()
    elif activation == 'leakyrelu':
        return nn.LeakyReLU(negative_slope=0.2)
    elif activation == 'tanh':
        return nn.Tanh()
    else:
        raise NotImplementedError(f"Activation {activation} is not supported yet.")

def get_norm(norm, in_channels, groups=32):
    if norm == 'none':
        return None
    elif norm == 'batchnorm':
        return nn.BatchNorm2d(in_channels)
    elif norm == 'groupnorm':
        return nn.GroupNorm(groups, in_channels)
    elif norm == 'instancenorm':
        return nn.InstanceNorm2d(in_channels)
    else:
        raise NotImplementedError(f"Norm {norm} is not supported yet.")

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 norm='batchnorm', activation='leakyrelu', transposed=False, bias=False):
        super().__init__()
        if transposed:
            self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)
        else:
            self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias)

        self.norm = get_norm(norm, out_channels)
        self.activation = get_activation(activation)

        self.init_weight()

    def init_weight(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.normal_(m.weight, 0, 0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def forward(self, x):
        x = self.conv(x)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x