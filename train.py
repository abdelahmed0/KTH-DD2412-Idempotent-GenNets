import argparse
import copy
import os
import torch
from tqdm import tqdm
import yaml

from model.dcgan import DCGAN
from util.dataset import load_mnist, load_celeb_a

from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter


def setup_dirs(config):
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(f, f_copy, opt, data_loader, config, device, writer):
    """Train the Idempotent Generative Network with optional Manifold Expansion Warmup"""

    n_epochs = config['training']['n_epochs']
    lambda_rec = config['losses']['lambda_rec']
    lambda_idem = config['losses']['lambda_idem']
    lambda_tight_end = config['losses']['lambda_tight']
    tight_clamp_ratio = config['losses']['tight_clamp_ratio']
    save_period = config['training']['save_period']
    run_id = config['run_id']

    # Manifold Warmup Parameters
    warmup_config = config['training'].get('manifold_warmup', {})
    warmup_enabled = warmup_config.get('enabled', False)
    warmup_epochs = warmup_config.get('warmup_epochs', 0)
    lambda_tight_start = warmup_config.get('lambda_tight_start', lambda_tight_end)
    schedule_type = warmup_config.get('schedule_type', 'linear')

    f.train()
    f_copy.eval()

    for epoch in tqdm(range(n_epochs), position=1, desc="Epoch"):
        # Calculate current lambda_tight based on warmup schedule
        if warmup_enabled and epoch < warmup_epochs:
            if schedule_type == "linear":
                lambda_tight = lambda_tight_start + (lambda_tight_end - lambda_tight_start) * (epoch / warmup_epochs)
            elif schedule_type == "exponential":
                lambda_tight = lambda_tight_start * (lambda_tight_end / lambda_tight_start) ** (epoch / warmup_epochs)
            else:
                raise ValueError(f"Unsupported schedule_type: {schedule_type}")
        else:
            lambda_tight = lambda_tight_end

        for batch_idx, (x, _) in enumerate(tqdm(data_loader, total=len(data_loader), position=0, desc="Train Step")):
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
            #tqdm.write(f"Epoch [{epoch+1}/{n_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss_item:.3f}")
            update_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Loss/Total', loss_item, update_step)
            writer.add_scalar('Loss/Reconstruction', loss_rec.item(), update_step)
            writer.add_scalar('Loss/Idempotence', loss_idem.item(), update_step)
            writer.add_scalar('Loss/Tightness', loss_tight.item(), update_step)
            writer.add_scalar('Hyperparameters/Lambda_Tight', lambda_tight, update_step)

        if (epoch + 1) % save_period == 0 or (epoch + 1) == n_epochs:
            checkpoint_path = os.path.join(config['checkpoint']['save_dir'], f"{run_id}_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': getattr(f, '_orig_mod', f).state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'loss': loss_item,
                'config': config
            }, checkpoint_path)
            tqdm.write(f"Saved checkpoint at epoch {epoch+1} \nEpoch [{epoch+1}/{n_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss_item:.3f}")


def main():
    """Usage: python train.py --config config.yaml"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Idempotent Generative Networks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    run_id = config['run_id']

    # Setup directories
    setup_dirs(config)

    # Load dataset
    dataset_name = config['dataset']['name']
    if dataset_name.lower() == "mnist":
        data_loader = load_mnist(batch_size=config['training']['batch_size'],
                                 download=config['dataset']['download'],
                                 num_workers=config['dataset']['num_workers'],
                                 pin_memory=config['dataset']['pin_memory'],
                                 single_channel=config['dataset']['single_channel'])
    elif dataset_name.lower() == "celeba":
        data_loader = load_celeb_a(batch_size=config['training']['batch_size'],
                                 download=config['dataset']['download'],
                                 num_workers=config['dataset']['num_workers'],
                                 pin_memory=config['dataset']['pin_memory'],
                                 split='train')
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported yet.")

    # Initialize TensorBoard writer
    log_dir = os.path.join(config['logging']['log_dir'], run_id)
    writer = SummaryWriter(log_dir=log_dir)

    # Setup device
    device = torch.device("cuda" if config['device']['use_cuda'] and torch.cuda.is_available() else "cpu")

    # Initialize models
    model = DCGAN(config['model']['architecture'])
    model_copy = copy.deepcopy(model).requires_grad_(False)

    model.to(device)
    model_copy.to(device)

    # Compile models if specified
    if config['training'].get('compile_model', False):
        model = torch.compile(model, mode="reduce-overhead")
        model_copy = torch.compile(model_copy, mode="reduce-overhead")

    # Initialize optimizer
    optimizer_config = config['optimizer']
    if optimizer_config['type'].lower() == "adam":
        optimizer = torch.optim.Adam(model.parameters(),
                                     lr=optimizer_config['lr'],
                                     betas=optimizer_config['betas'])
    else:
        raise NotImplementedError(f"Optimizer type {optimizer_config['type']} is not supported.")

    try:
        train(
            f=model,
            f_copy=model_copy,
            opt=optimizer,
            data_loader=data_loader,
            config=config,
            device=device,
            writer=writer
        )
    except KeyboardInterrupt:
        print("Training interrupted.")
    finally:
        writer.close()


if __name__ == "__main__":
    main()
