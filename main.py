import torch

from dcgan import DCGAN
from torch.utils.data import DataLoader
from torchvision.datasets.mnist import MNIST


def train(f, f_copy, opt, data_loader, n_epochs):
    for epoch in range(n_epochs):
        for x in data_loader:
            z = torch.randn_like(x)

            # Apply f to get all needed
            f_copy.load_state_dict(f.state_dict())
            fx = f(x)
            fz = f(z)
            f_z = fz.detach()
            ff_z = f_copy(f_z)
            f_fz = f_copy(fz)

            # Calculate losses
            loss_rec = (fx - x).pow(2).mean()
            loss_idem = (f_fz - fz).pow(2).mean()
            loss_tight = -(ff_z - f_z).pow(2).mean()

            # Optimize for losses
            loss = loss_rec + loss_idem + loss_tight * 0.1
            opt.zero_grad()
            loss.backward()
            opt.step()

            print(f"Epoch {epoch}, Loss: {loss.item()}")

# Load MNIST
mnist = torch.utils.data.DataLoader(
    MNIST(root="./data", download=True),
    batch_size=256,
    shuffle=True
)

# TODO: Tensorboard logging
n_epochs = 2 # TODO: Set to 1000 as in the paper later
batch_size = 256
model = DCGAN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

train(model, model.copy(), optimizer, mnist, n_epochs)