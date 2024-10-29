import argparse
import copy
import os
import torch
from tqdm import tqdm

from dcgan import DCGAN
from dataset import load_mnist

from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter



def setup_dirs():
    os.makedirs("data", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("runs", exist_ok=True)


def train(f, f_copy, opt, data_loader, n_epochs, device, writer, run_id,
          lambda_rec=20, lambda_idem=20, lambda_tight=2.5, tight_clamp_ratio=1.5,
          save_period=10):
    
    f.train()
    f_copy.eval()

    for epoch in tqdm(range(n_epochs), position=1, desc="Epoch"):
        for batch_idx, (x, _) in (tqdm_bar := tqdm(enumerate(data_loader), total=len(data_loader), position=0, desc="Train Step")):
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
            loss_tight = F.l1_loss(ff_z, f_z)

            # Smoothen tightness loss
            loss_tight = torch.tanh(loss_tight / (tight_clamp_ratio * loss_rec)) * tight_clamp_ratio * loss_rec
            
            # Optimize for losses
            loss = lambda_rec * loss_rec + lambda_idem * loss_idem - lambda_tight * loss_tight
            opt.zero_grad()
            loss.backward()
            opt.step()

            loss_item = loss.item()
            tqdm_bar.set_postfix_str(f'Loss: {loss_item:0.3f}')
            update_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Loss/Total', loss_item, update_step)
            writer.add_scalar('Loss/Reconstruction', loss_rec.item(), update_step)
            writer.add_scalar('Loss/Idempotence', loss_idem.item(), update_step)
            writer.add_scalar('Loss/Tightness', loss_tight.item(), update_step)

        
        if epoch % save_period == 0 or epoch + 1 == n_epochs:
            torch.save({
                'epoch': epoch,
                'model_state_dict': getattr(f, '_orig_mod', f).state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss
            }, f"checkpoints/{run_id}_{epoch}.pt")
            tqdm_bar.write(f"Saved checkpoint at epoch {epoch}")


if __name__ == "__main__":
    """Usage: python train.py --run_id <run_id>"""
    # TODO: Implement real-data related noise mentioned at the end of chapter 2
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()

    run_id = args.run_id

    # Hyperparameters
    n_epochs = 1000
    batch_size = 256
    save_period = 100
    compile_model = True

    # Setup
    mnist = load_mnist(batch_size=batch_size)
    setup_dirs()

    writer = SummaryWriter(log_dir=f'runs/{run_id}')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = DCGAN()
    model_copy = copy.deepcopy(model).requires_grad_(False)

    model.to(device)
    model_copy.to(device)

    if compile_model:
        model = torch.compile(model, mode="reduce-overhead")
        model_copy = torch.compile(model_copy, mode="reduce-overhead")

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, betas=(0.5, 0.999))

    try:
        train(model, model_copy, optimizer, mnist, n_epochs, device, writer, run_id, save_period=save_period)
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        writer.close()