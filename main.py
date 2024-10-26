import copy
import torch

from dcgan import DCGAN
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


def load_mnist():
    transform = transforms.Compose([
    transforms.Resize(64),  
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
    ])

    mnist = DataLoader(
        MNIST(root="./data", download=True, transform=transform),
        batch_size=256,
        shuffle=True,
        num_workers=4
    )

    return mnist

def train(f, f_copy, opt, data_loader, n_epochs, device, writer=None,
          lambda_rec=20, lambda_idem=20, lambda_tight=2.5, tight_clamp_ratio=1.5):
    
    f.train()

    for epoch in range(n_epochs):
        for batch_idx, (x, _) in enumerate(data_loader):
            x = x.to(device)

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

            # Smoothen tightness loss
            loss_tight = torch.tanh(loss_tight / (tight_clamp_ratio * loss_rec)) * tight_clamp_ratio * loss_rec
            
            # Optimize for losses
            loss = lambda_rec * loss_rec + lambda_idem * loss_idem + lambda_tight * loss_tight
            opt.zero_grad()
            loss.backward()
            opt.step()

            if writer:
                update_step = epoch * len(data_loader) + batch_idx
                writer.add_scalar('Loss/Total', loss.item(), update_step)
                writer.add_scalar('Loss/Reconstruction', loss_rec.item(), update_step)
                writer.add_scalar('Loss/Idempotency', loss_idem.item(), update_step)
                writer.add_scalar('Loss/Tightness', loss_tight.item(), update_step)
            elif batch_idx % 100 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx} Loss: {loss.item()}")


if __name__ == "__main__":
    # TODO: Implement real-data related noise mentioned at the end of chapter 2
    mnist = load_mnist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_epochs = 1000
    batch_size = 256

    model = DCGAN()
    model_copy = copy.deepcopy(model).requires_grad_(False)

    model.to(device)
    model_copy.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

    writer = SummaryWriter(log_dir='runs/idempotent_gan')

    try:
        train(model, model_copy, optimizer, mnist, n_epochs, device, writer)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        writer.close()