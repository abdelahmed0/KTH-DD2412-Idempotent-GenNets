import argparse
import copy
import os
import torch
import yaml
import time

from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import GradScaler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from model.dcgan import DCGAN
from util.dataset import load_mnist, load_celeb_a
from util.function_util import fourier_sample
from util.model_util import load_checkpoint
from generate import rec_generate_images


def setup_dirs(config: dict):
    """Create necessary directories based on the configuration."""
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)


def load_config(config_path: str) -> dict:
    """Load YAML configuration from a file."""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def compute_losses(f: DCGAN, f_copy: DCGAN, x: torch.Tensor, z: torch.Tensor,
                  rec_func, idem_func, tight_func,
                  tight_clamp: bool, tight_clamp_ratio: float):
    """
    Compute reconstruction, idempotency, and tightness losses.

    Args:
        f (DCGAN): Primary model.
        f_copy (DCGAN): Copy of the primary model.
        x (torch.Tensor): Input tensor.
        z (torch.Tensor): Latent tensor.
        rec_func: Reconstruction loss function.
        idem_func: Idempotency loss function.
        tight_func: Tightness loss function.
        tight_clamp (bool): Whether to apply tightness clamp.
        tight_clamp_ratio (float): Ratio for tightness clamp.

    Returns:
        tuple: (loss_rec, loss_idem, loss_tight)
    """
    # Forward passes
    fx = f(x)
    fz = f(z)
    
    # Update f_copy to mirror f without tracking gradients
    f_copy.load_state_dict(f.state_dict())
    
    # Compute idempotency and tightness
    f_fz = f_copy(fz)   # Idempotency target
    ff_z = f(fz)        # Tightness target
    f_z = fz.detach()   # Detached for tightness loss
    
    # Calculate losses
    loss_rec = rec_func(fx, x)
    loss_idem = idem_func(f_fz, fz)
    loss_tight = -tight_func(ff_z, f_z)
    
    # Smoothen tightness loss
    if tight_clamp:
        loss_tight = torch.tanh(loss_tight / (tight_clamp_ratio * loss_rec)) * tight_clamp_ratio * loss_rec
    
    return loss_rec, loss_idem, loss_tight


def train(f: DCGAN, f_copy: DCGAN, opt: torch.optim.Optimizer, scaler: GradScaler,
          data_loader: DataLoader, val_data_loader: DataLoader, config: dict,
          device: torch.device, writer: SummaryWriter):
    """Train the Idempotent Generative Network with optional Manifold Expansion Warmup"""

    # Extract configurations
    n_epochs = config['training']['n_epochs']
    loss_function = config['losses']['loss_function']
    lambda_rec = config['losses']['lambda_rec']
    lambda_idem = config['losses']['lambda_idem']
    lambda_tight_end = config['losses']['lambda_tight']
    tight_clamp = config['losses']['tight_clamp']
    tight_clamp_ratio = config['losses']['tight_clamp_ratio']
    save_period = config['training']['save_period']
    image_log_period = config['training'].get('image_log_period', 100)
    validation_period = config['training'].get('validation_period', 1)
    run_id = config['run_id']
    use_fourier_sampling = config['training'].get('use_fourier_sampling', False)
    use_amp = config['training']['use_amp']
    use_validation = config['training'].get('use_validation', True)  # Added: use_validation flag

    # Early Stopping Parameters
    if use_validation:
        patience = config['early_stopping'].get('patience', 5)
        best_val_loss = float('inf')
        epochs_without_improvement = 0
    else:
        # Initialize variables to avoid reference before assignment
        patience = None
        best_val_loss = None
        epochs_without_improvement = None

    # Manifold Warmup Parameters
    warmup_config = config['training'].get('manifold_warmup', {})
    warmup_enabled = warmup_config.get('enabled', False)
    warmup_epochs = warmup_config.get('warmup_epochs', 0)
    lambda_tight_start = warmup_config.get('lambda_tight_start', lambda_tight_end)
    schedule_type = warmup_config.get('schedule_type', 'linear')

    # Define loss functions
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

    # Log configuration
    writer.add_text("config", f"``` {config} ```")

    for epoch in tqdm(range(config.get('start_epoch', 0), n_epochs),
                      position=1, desc="Epoch", total=n_epochs,
                      initial=config.get('start_epoch', 0)):
        epoch_timer = time.time()
        f.train()
        f_copy.train()

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

        for batch_idx, (x, _) in enumerate(tqdm(data_loader, total=len(data_loader),
                                               position=0, desc="Train Step")):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                x = x.to(device)

                # Sample z
                if use_fourier_sampling:
                    z = fourier_sample(x)
                else:
                    z = torch.randn_like(x, device=device)

                # Compute losses using helper function
                loss_rec, loss_idem, loss_tight = compute_losses(
                    f=f,
                    f_copy=f_copy,
                    x=x,
                    z=z,
                    rec_func=rec_func,
                    idem_func=idem_func,
                    tight_func=tight_func,
                    tight_clamp=tight_clamp,
                    tight_clamp_ratio=tight_clamp_ratio
                )

                # Combine losses
                loss = lambda_rec * loss_rec + lambda_idem * loss_idem + lambda_tight * loss_tight

            # Backpropagation and optimization
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad()

            # Logging
            loss_item = loss.item()
            update_step = epoch * len(data_loader) + batch_idx
            writer.add_scalar('Loss/Total', loss_item, update_step)
            writer.add_scalar('Loss/Reconstruction', loss_rec.item(), update_step)
            writer.add_scalar('Loss/Idempotence', loss_idem.item(), update_step)
            writer.add_scalar('Loss/Tightness', loss_tight.item(), update_step)
            writer.add_scalar('Hyperparameters/Lambda_Tight', lambda_tight, update_step)

        # Update current epoch in config
        config['current_epoch'] = epoch  # Used when terminating training
        writer.add_scalar('Logs/Epoch_Timer', time.time() - epoch_timer, epoch+1)

        # Validation
        if use_validation and ((epoch + 1) % validation_period == 0 or (epoch + 1) == n_epochs):
            # Initialize accumulators
            val_loss_rec = 0.0
            val_loss_idem = 0.0
            val_loss_tight = 0.0
            val_batches = 0

            # Set models to evaluation mode
            f.eval()
            f_copy.eval()

            with torch.no_grad():
                for x_val, _ in tqdm(val_data_loader, desc="Validation", leave=False):
                    x_val = x_val.to(device)

                    # Sample z_val similarly to training
                    if use_fourier_sampling:
                        z_val = fourier_sample(x_val)
                    else:
                        z_val = torch.randn_like(x_val, device=device)

                    # Compute losses using helper function
                    loss_rec_val, loss_idem_val, loss_tight_val = compute_losses(
                        f=f,
                        f_copy=f_copy,
                        x=x_val,
                        z=z_val,
                        rec_func=rec_func,
                        idem_func=idem_func,
                        tight_func=tight_func,
                        tight_clamp=tight_clamp,
                        tight_clamp_ratio=tight_clamp_ratio
                    )

                    # Accumulate losses
                    val_loss_rec += loss_rec_val.item()
                    val_loss_idem += loss_idem_val.item()
                    val_loss_tight += loss_tight_val.item()
                    val_batches += 1

            # Calculate average losses
            avg_val_loss_rec = val_loss_rec / val_batches
            avg_val_loss_idem = val_loss_idem / val_batches
            avg_val_loss_tight = val_loss_tight / val_batches
            avg_val_total_loss = lambda_rec * avg_val_loss_rec + lambda_idem * avg_val_loss_idem + lambda_tight * avg_val_loss_tight

            # Log validation losses
            writer.add_scalar('Loss/Validation_Total', avg_val_total_loss, epoch+1)
            writer.add_scalar('Loss/Validation_Reconstruction', avg_val_loss_rec, epoch+1)
            writer.add_scalar('Loss/Validation_Idempotence', avg_val_loss_idem, epoch+1)
            writer.add_scalar('Loss/Validation_Tightness', avg_val_loss_tight, epoch+1)
            tqdm.write(f"Epoch [{epoch+1}/{n_epochs}], Validation Losses -> "
                       f"Total: {avg_val_total_loss:.4f}, "
                       f"Reconstruction: {avg_val_loss_rec:.4f}, "
                       f"Idempotence: {avg_val_loss_idem:.4f}, "
                       f"Tightness: {avg_val_loss_tight:.4f}")

            # Early Stopping Check based on total validation loss
            if avg_val_total_loss < best_val_loss:
                best_val_loss = avg_val_total_loss
                epochs_without_improvement = 0
                # Save the best model
                checkpoint_path = os.path.join(config['checkpoint']['save_dir'], f"{run_id}_best_model.pt")
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': getattr(f, '_orig_mod', f).state_dict(),
                    'optimizer_state_dict': opt.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'loss': avg_val_total_loss,
                    'config': config
                }, checkpoint_path)
                tqdm.write(f"Validation loss improved to {avg_val_total_loss:.4f}, saved best model.")
            else:
                epochs_without_improvement += 1
                tqdm.write(f"Validation loss did not improve for {epochs_without_improvement} epochs.")
                if epochs_without_improvement >= patience:
                    tqdm.write(f"Early stopping triggered after {patience} epochs without improvement.")
                    break  # Break out of the training loop

            # Reset models to training mode
            f.train()
            f_copy.train()

        # Image Logging
        if (epoch + 1) % image_log_period == 0 or (epoch + 1) == n_epochs:
            # Save the sampled image
            f.eval()
            original, reconstructed = rec_generate_images(
                model=f, device=device, data=data_loader,
                n_images=5, n_recursions=1,
                reconstruct=True, use_fourier_sampling=use_fourier_sampling
            )
            noise, generated = rec_generate_images(
                model=f, device=device, data=data_loader,
                n_images=5, n_recursions=1,
                reconstruct=False, use_fourier_sampling=use_fourier_sampling
            )
            f.train()

            writer.add_images('Image/Generated', generated[:, 0].detach(), epoch+1)
            writer.add_images('Image/Noise', noise.detach(), epoch+1)
            writer.add_images('Image/Reconstructed', reconstructed[:, 0].detach(), epoch+1)
            writer.add_images('Image/Original', original.detach(), epoch+1)

            tqdm.write(f"Logged images for epoch [{epoch+1}/{n_epochs}]")

        # Checkpoint Saving
        if (epoch + 1) % save_period == 0 or (epoch + 1) == n_epochs:
            checkpoint_path = os.path.join(config['checkpoint']['save_dir'], f"{run_id}_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': getattr(f, '_orig_mod', f).state_dict(),
                'optimizer_state_dict': opt.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': loss_item,
                'config': config
            }, checkpoint_path)
            tqdm.write(f"Saved checkpoint at epoch {epoch+1} \n"
                       f"Epoch [{epoch+1}/{n_epochs}], Batch [{batch_idx+1}/{len(data_loader)}], Loss: {loss_item:.3f}")


def main():
    """Usage: python train.py --config config.yaml"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Idempotent Generative Networks")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from the specified checkpoint")
    args = parser.parse_args()

    for config_path in ["config_mnist_fourier.yaml",
                        "config_mnist.yaml",  
                        "config_mnist_ignite.yaml",
                        "config_mnist_clamp.yaml"]:

        # Load configuration
        if args.resume is None:
            config = load_config(config_path)
        else:
            checkpoint = load_checkpoint(args.resume)
            config = checkpoint['config']
            config['start_epoch'] = checkpoint['epoch']
        run_id = config['run_id']

        # Setup directories
        setup_dirs(config)

        # Determine if validation is used
        use_validation = config['training'].get('use_validation', True)

        # Load dataset
        dataset_name = config['dataset']['name']
        if dataset_name.lower() == "mnist":
            if use_validation:
                # Modify load_mnist to return both train and validation loaders
                train_loader, val_loader = load_mnist(
                    batch_size=config['training']['batch_size'],
                    download=config['dataset']['download'],
                    num_workers=config['dataset']['num_workers'],
                    pin_memory=config['dataset']['pin_memory'],
                    single_channel=config['dataset']['single_channel'],
                    validation_split=config['dataset'].get('validation_split', 0.1),
                    add_noise=config['dataset'].get('add_noise', False)
                )
            else:
                # Only load training data
                train_loader = load_mnist(
                    batch_size=config['training']['batch_size'],
                    download=config['dataset']['download'],
                    num_workers=config['dataset']['num_workers'],
                    pin_memory=config['dataset']['pin_memory'],
                    single_channel=config['dataset']['single_channel'],
                    validation_split=0.0,  # No validation split
                    add_noise=config['dataset'].get('add_noise', False)
                )
                val_loader = None
        elif dataset_name.lower() == "celeba":
            if use_validation:
                # For CelebA, use 'train' and 'valid' splits
                train_loader = load_celeb_a(
                    batch_size=config['training']['batch_size'],
                    download=config['dataset']['download'],
                    num_workers=config['dataset']['num_workers'],
                    pin_memory=config['dataset']['pin_memory'],
                    split='train'
                )
                val_loader = load_celeb_a(
                    batch_size=config['training']['batch_size'],
                    download=config['dataset']['download'],
                    num_workers=config['dataset']['num_workers'],
                    pin_memory=config['dataset']['pin_memory'],
                    split='valid'
                )
            else:
                # Only load training data
                train_loader = load_celeb_a(
                    batch_size=config['training']['batch_size'],
                    download=config['dataset']['download'],
                    num_workers=config['dataset']['num_workers'],
                    pin_memory=config['dataset']['pin_memory'],
                    split='train'
                )
                val_loader = None
        else:
            raise NotImplementedError(f"Dataset {dataset_name} is not supported yet.")

        # Initialize TensorBoard writer
        log_dir = os.path.join(config['logging']['log_dir'], run_id)
        writer = SummaryWriter(log_dir=log_dir)

        # Setup device
        device = torch.device("cuda" if config['device']['use_cuda'] and torch.cuda.is_available() else "cpu")

        # Initialize models
        model = DCGAN(architecture=config['model']['architecture'], use_bias=config['model']['use_bias'])
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

        scaler = GradScaler(device.type, enabled=config['training']['use_amp'])

        if args.resume is not None and use_validation:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

        torch.backends.cudnn.benchmark = True 

        try:
            train(
                f=model,
                f_copy=model_copy,
                opt=optimizer,
                scaler=scaler,
                data_loader=train_loader,
                val_data_loader=val_loader if use_validation else None,
                config=config,
                device=device,
                writer=writer
            )
        except KeyboardInterrupt:
            print("Training interrupted.")
            ans = input("Do you want to save a checkpoint (Y/N)?")
            if ans.lower() in ['yes', 'y']:
                if use_validation:
                    epoch_to_save = config['current_epoch'] + 1
                else:
                    epoch_to_save = config.get('current_epoch', 0) + 1
                checkpoint_path = os.path.join(config['checkpoint']['save_dir'], f"{run_id}_epoch_{epoch_to_save}.pt")
                torch.save({
                    'epoch': epoch_to_save,
                    'model_state_dict': getattr(model, '_orig_mod', model).state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scaler_state_dict': scaler.state_dict(),
                    'config': config
                }, checkpoint_path)
                print("Saved model to: ", checkpoint_path)
        finally:
            writer.close()


if __name__ == "__main__":
    main()
