import argparse
import gc
import numpy as np
import os
import torch
import yaml

from torch.utils.tensorboard.writer import SummaryWriter

from model.dcgan import DCGAN
from model.u_net_conditional import UNetConditional
from model.u_net import UNet
from trainer.ign_conditional_trainer import IGNConditionalTrainer
from trainer.ign_trainer import IGNTrainer
from util.dataset import load_mnist, load_celeb_a
from util.model_util import load_checkpoint


def setup_dirs(config):
    os.makedirs(config['checkpoint']['save_dir'], exist_ok=True)
    os.makedirs(config['logging']['log_dir'], exist_ok=True)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config


def get_trainer(args, config: dict, checkpoint: dict, device: torch.device):
    # Model params
    architecture = config["model"]["architecture"]
    norm = config["model"]["norm"]
    use_bias = config["model"]["use_bias"]
    input_size = config["model"].get("input_size", 64)
    num_groups = config["model"].get("num_groups", 32)
    dropout = config["model"].get("dropout", None)

    # Initialize models
    if "dcgan" in architecture.lower():
        model = DCGAN(
            architecture=architecture,
            input_size=input_size,
            norm=norm,
            use_bias=use_bias,
            num_groups=num_groups,
            dropout=dropout,
        )
        trainer = IGNTrainer(model=model, config=config, device=device, checkpoint=checkpoint)
    elif "unet" in architecture.lower().replace("_",""):
        if "conditional" in architecture.lower():
            model = UNetConditional(device)
            trainer = IGNConditionalTrainer(model=model, config=config, device=device, checkpoint=checkpoint)
        else:
            model = UNet()
            trainer = IGNTrainer(model=model, config=config, device=device, checkpoint=checkpoint)

    return trainer

def main():
    """Usage: python train.py --config config.yaml"""
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train Idempotent Generative Networks")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to the config file")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from the specified checkpoint")
    parser.add_argument("--seed", type=int, default=None, help="Set pytorch seed")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Load configuration
    if args.resume is None:
        config = load_config(args.config)
        checkpoint = None
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
    elif dataset_name.lower() == "celeba":
        # For CelebA, use 'train' and 'valid' splits
        train_loader = load_celeb_a(
            batch_size=config['training']['batch_size'],
            download=config['dataset']['download'],
            num_workers=config['dataset']['num_workers'],
            pin_memory=config['dataset']['pin_memory'],
            random_flip=config['dataset'].get('random_flip', False),
            split='train'
        )
        val_loader = load_celeb_a(
            batch_size=config['training']['batch_size'],
            download=config['dataset']['download'],
            num_workers=config['dataset']['num_workers'],
            pin_memory=config['dataset']['pin_memory'],
            random_flip=config['dataset'].get('random_flip', False),
            split='valid'
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported yet.")

    # Initialize TensorBoard writer
    log_dir = os.path.join(config['logging']['log_dir'], run_id)
    writer = SummaryWriter(log_dir=log_dir)

    # Setup device
    device = torch.device("cuda" if config['device']['use_cuda'] and torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        print("Set high matmul precision!")
        torch.set_float32_matmul_precision('high')

    torch._dynamo.config.cache_size_limit = 16 # Graph breaks too often, debug using "TORCH_LOGS="recompiles" python train_conditional.py --config config.yaml"
    torch.backends.cudnn.benchmark = True 

    try:
        completed = False
        while not completed:
            trainer = get_trainer(args, config, checkpoint, device)

            completed = trainer.fit(
                data_loader=train_loader,
                val_data_loader=val_loader,
                writer=writer
            )

            del trainer
            gc.collect()
            torch.cuda.empty_cache()
            torch.clear_autocast_cache()
    except KeyboardInterrupt:
        print("Training interrupted.")
        ans = input("Do you want to save a checkpoint (Y/N)?")
        if ans.lower() in ['yes', 'y']:
            checkpoint_path = os.path.join(config['checkpoint']['save_dir'], f"{run_id}_epoch_{trainer.config['current_epoch']+1}.pt")
            trainer.save_model(trainer.config['current_epoch'] + 1, checkpoint_path)
            print("Saved model to: ", checkpoint_path)
    finally:
        writer.close()


if __name__ == "__main__":
    main()
