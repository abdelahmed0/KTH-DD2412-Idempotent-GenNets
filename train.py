import argparse
import copy
import os
import torch
import yaml

from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from model.dcgan import DCGAN
from util.dataset import load_mnist, load_celeb_a
from util.function_util import fourier_sample
from util.model_util import load_checkpoint
from generate import rec_generate_images


def setup_dirs(config):
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def train(f: DCGAN, f_copy: DCGAN, opt: torch.optim.Optimizer, data_loader: DataLoader, config: dict, device: str, writer: SummaryWriter):
    """Train the Idempotent Generative Network with optional Manifold Expansion Warmup"""

    n_epochs = config['training']['n_epochs']
    loss_function = config['losses']['loss_function']
    lambda_rec = config['losses']['lambda_rec']
    lambda_idem = config['losses']['lambda_idem']
    lambda_tight_end = config['losses']['lambda_tight']
    tight_clamp_ratio = config['losses']['tight_clamp_ratio']
    save_period = config['training']['save_period']
    image_log_period = config['training'].get('image_log_period', 100)
    run_id = config['run_id']
    use_fourier_sampling = config['training'].get('use_fourier_sampling', False)

    # Manifold Warmup Parameters
    warmup_config = config['training'].get('manifold_warmup', {})
    warmup_enabled = warmup_config.get('enabled', False)
    warmup_epochs = warmup_config.get('warmup_epochs', 0)
    lambda_tight_start = warmup_config.get('lambda_tight_start', lambda_tight_end)
    schedule_type = warmup_config.get('schedule_type', 'linear')

    # Loss function
    if loss_function.lower() == "l1":
        rec_func = F.l1_loss
        idem_func = F.l1_loss
        tight_func = F.l1_loss
    elif loss_function.lower() == "mse":
        rec_func = F.mse_loss
        idem_func = F.mse_loss
        tight_func = F.mse_loss
    else:
        raise NotImplementedError(f"Loss function '{loss_function}' is not supported yet.") 
    
    writer.add_text("config", f"``` {config} ```")

    f.train()
    f_copy.eval()

    for epoch in tqdm(range(config.get('start_epoch', 0), n_epochs), position=1, desc="Epoch", total=n_epochs, initial=config.get('start_epoch', 0)):
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

            if use_fourier_sampling:
                z = fourier_sample(x)
                if epoch == 1 and batch_idx == 1:
                    # Save the sampled image
                    writer.add_image('Image/Sampled', z[0], 0)
            else:
                z = torch.randn_like(x, device=device)

            # Apply f to get all needed
            f_copy.load_state_dict(f.state_dict())
            fx = f(x)
            fz = f(z)
            f_z = fz.detach()
            ff_z = f(f_z)
            f_fz = f_copy(fz)

            # Calculate losses
            loss_rec = rec_func(fx, x)
            loss_idem = idem_func(f_fz, fz)
            loss_tight = tight_func(ff_z, f_z)

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

        if (epoch + 1) % image_log_period == 0 or (epoch + 1) == n_epochs:
            # Save the sampled image
            original, reconstructed = rec_generate_images(model=f, device=device, data=data_loader, n_images=1, n_recursions=1, reconstruct=True, use_fourier_sampling=use_fourier_sampling)
            noise, generated = rec_generate_images(model=f, device=device, data=data_loader, n_images=1, n_recursions=1, reconstruct=False, use_fourier_sampling=use_fourier_sampling)

            writer.add_images('Image/Generated', generated[0][0][:5].detach(), epoch+1)
            writer.add_images('Image/Noise', noise[0][:5].detach(), epoch+1)
            writer.add_images('Image/Reconstructed', reconstructed[0][0][:5].detach(), epoch+1)
            writer.add_images('Image/Original', original[0][:5].detach(), epoch+1)
            tqdm.write(f"Logged images for epoch [{epoch+1}/{n_epochs}]")

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
    parser.add_argument("--resume", type=str, default=None, help="Resume training from the specified checkpoint")
    args = parser.parse_args()

    # Load configuration
    if args.resume is None:
        config = load_config(args.config)
    else:
        checkpoint = load_checkpoint(args.resume)
        config = checkpoint['config']
        config['start_epoch'] = checkpoint['epoch']
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
    if args.resume is not None:
        model.load_state_dict(checkpoint['model_state_dict'])
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
    
    if args.resume is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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
