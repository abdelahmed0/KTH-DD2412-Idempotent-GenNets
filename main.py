import copy
import torch

from dcgan import DCGAN
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


def load_mnist():
    transform = transforms.Compose([
    transforms.Resize(64),  
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
    ])

    mnist = torch.utils.data.DataLoader(
        MNIST(root="./data", download=True, transform=transform),
        batch_size=256,
        shuffle=True,
        num_workers=4
    )

    return mnist

def train(f, f_copy, opt, data_loader, n_epochs, device,
          lambda_rec=20, lambda_idem=20, lambda_tight=2.5):
    
    f.train()

    for epoch in range(n_epochs):
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)

            # TODO: Validate if this is the correct way to apply latent sampling
            z = torch.randn_like(x, device=device)

            # Apply f to get all needed
            f_copy.load_state_dict(f.state_dict())
            fx = f(x)
            fz = f(z)
            f_z = fz.detach()
            ff_z = f_copy(f_z)
            f_fz = f_copy(fz)

            # Calculate losses
            loss_rec = F.l1_loss(fx, x)
            loss_idem = F.l1_loss(f_fz, fz)
            loss_tight = -F.l1_loss(ff_z, f_z)
            
            # Optimize for losses
            loss = lambda_rec * loss_rec + lambda_idem * loss_idem + lambda_tight * loss_tight
            opt.zero_grad()
            loss.backward()
            opt.step()

            if batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx} Loss: {loss.item()}")


mnist = load_mnist()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# TODO: Tensorboard logging
n_epochs = 2 # TODO: Set to 1000 as in the paper later
batch_size = 256

model = DCGAN()
model_copy = copy.deepcopy(model).requires_grad_(False)

model.to(device)
model_copy.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

train(model, model_copy, optimizer, mnist, n_epochs, device)
