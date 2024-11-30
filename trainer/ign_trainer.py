import copy
import os
import time
import torch

from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from model.dcgan import DCGAN
from util.function_util import fourier_sample, normalize_batch
from generate import rec_generate_images
from util.scoring import evaluate_generator


class IGNTrainer:
    def __init__(
        self, 
        model: torch.nn.Module, 
        config: dict, 
        checkpoint: dict = None, 
        device=torch.device("cpu"),
    ):
        self.config = config
        self.device = device
        
        self.model = model
        self.model_copy = copy.deepcopy(self.model).requires_grad_(False)

        # Initialize optimizer
        optimizer_config = config["optimizer"]
        if optimizer_config["type"].lower() == "adam":
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=optimizer_config["lr"],
                betas=optimizer_config["betas"],
            )
        else:
            raise NotImplementedError(
                f"Optimizer type {optimizer_config['type']} is not supported."
            )

        self.scaler = torch.GradScaler(
            device.type, enabled=config["training"]["use_amp"]
        )

        self.model = self.model.to(device)
        self.model_copy = self.model_copy.to(device)

        if checkpoint is not None:
            self.load_model(checkpoint)

        # Compile models if specified
        if config["training"].get("compile_model", False):
            self.compile()

        # Loss function
        loss_function = self.config["losses"]["loss_function"]
        if loss_function.lower() == "l1":
            self.rec_func = F.l1_loss
            self.idem_func = F.l1_loss
            self.tight_func = F.l1_loss
        elif loss_function.lower() == "mse":
            self.rec_func = F.mse_loss
            self.idem_func = F.mse_loss
            self.tight_func = F.mse_loss
        else:
            raise NotImplementedError(
                f"Loss function '{loss_function}' is not supported yet."
            )

    def compile(self):
        self.model = torch.compile(self.model, mode="reduce-overhead")
        self.model_copy = torch.compile(self.model_copy, mode="reduce-overhead")

    def load_model(self, checkpoint: dict):
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

    def save_model(self, epoch: int, path: str):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": getattr(
                    self.model, "_orig_mod", self.model
                ).state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scaler_state_dict": self.scaler.state_dict(),
                "config": self.config,
            },
            path,
        )

    def get_losses(
        self,
        x: torch.Tensor,
        lambda_rec: float,
        lambda_idem: float,
        lambda_tight: float,
        tight_clamp: bool,
        tight_clamp_ratio: float,
    ):
        use_fourier_sampling = self.config["training"].get(
            "use_fourier_sampling",
            False,
        )
        batch_size = x.shape[0]

        if use_fourier_sampling:
            z = fourier_sample(x)
        else:
            z = torch.randn_like(x)

        self.model_copy.load_state_dict(self.model.state_dict())
        fx = self.model(x)
        fz = self.model(z)
        f_z = fz.detach()
        ff_z = self.model(f_z)
        f_fz = self.model_copy(fz)

        # Calculate losses
        loss_rec = (
            self.rec_func(fx, x, reduction="none").reshape(batch_size, -1).mean(dim=1)
        )
        loss_idem = self.idem_func(f_fz, fz, reduction="mean")
        loss_tight = (
            -self.tight_func(ff_z, f_z, reduction="none")
            .reshape(batch_size, -1)
            .mean(dim=1)
        )

        # Smoothen tightness loss
        if tight_clamp:
            loss_tight = (
                F.tanh(loss_tight / (tight_clamp_ratio * loss_rec))
                * tight_clamp_ratio
                * loss_rec
            )

        # Calculate means
        loss_rec = loss_rec.mean()
        loss_tight = loss_tight.mean()

        # Optimize for losses
        total_loss = (
            lambda_rec * loss_rec + lambda_idem * loss_idem + lambda_tight * loss_tight
        )

        return total_loss, loss_rec, loss_idem, loss_tight

    def _validate(self, data_loader: DataLoader, epoch: int):
        # Validation after epoch
        lambda_rec = self.config["losses"]["lambda_rec"]
        lambda_idem = self.config["losses"]["lambda_idem"]
        lambda_tight_end = self.config["losses"]["lambda_tight"]
        tight_clamp = self.config["losses"]["tight_clamp"]
        tight_clamp_ratio = self.config["losses"]["tight_clamp_ratio"]

        # Calculate current lambda_tight based on warmup schedule
        warmup_config = self.config["training"].get("manifold_warmup", {})
        warmup_enabled = warmup_config.get("enabled", False)
        warmup_epochs = warmup_config.get("warmup_epochs", 0)
        lambda_tight_start = warmup_config.get("lambda_tight_start", lambda_tight_end)
        schedule_type = warmup_config.get("schedule_type", "linear")
        if warmup_enabled and epoch < warmup_epochs:
            if schedule_type == "linear":
                lambda_tight = lambda_tight_start + (
                    lambda_tight_end - lambda_tight_start
                ) * (epoch / warmup_epochs)
            elif schedule_type == "exponential":
                lambda_tight = lambda_tight_start * (
                    lambda_tight_end / lambda_tight_start
                ) ** (epoch / warmup_epochs)
            else:
                raise ValueError(f"Unsupported schedule_type: {schedule_type}")
        else:
            lambda_tight = lambda_tight_end

        val_loss_rec = 0.0
        val_loss_idem = 0.0
        val_loss_tight = 0.0
        val_loss_total = 0.0
        val_batches = 0
        with torch.inference_mode():
            for x, _ in data_loader:
                torch.compiler.cudagraph_mark_step_begin()
                x = x.to(self.device)

                total_loss, loss_rec, loss_idem, loss_tight = self.get_losses(
                    x=x,
                    lambda_rec=lambda_rec,
                    lambda_idem=lambda_idem,
                    lambda_tight=lambda_tight,
                    tight_clamp=tight_clamp,
                    tight_clamp_ratio=tight_clamp_ratio,
                )

                val_loss_rec += loss_rec.item()
                val_loss_idem += loss_idem.item()
                val_loss_tight += loss_tight.item()
                val_loss_total += total_loss.item()
                val_batches += 1

        # Calculate average losses
        avg_val_loss_rec = val_loss_rec / val_batches
        avg_val_loss_idem = val_loss_idem / val_batches
        avg_val_loss_tight = val_loss_tight / val_batches
        avg_val_total_loss = val_loss_total / val_batches

        return avg_val_total_loss, avg_val_loss_rec, avg_val_loss_idem, avg_val_loss_tight

    def log_validation(self, data_loader: DataLoader, writer: SummaryWriter, epoch: int):
        # In train mode
        self.model.train()
        self.model_copy.train()
        avg_val_total_loss, avg_val_loss_rec, avg_val_loss_idem, avg_val_loss_tight = self._validate(data_loader, epoch)

        writer.add_scalar("Validation_Train/Total", avg_val_total_loss, epoch + 1)
        writer.add_scalar("Validation_Train/Reconstruction", avg_val_loss_rec, epoch + 1)
        writer.add_scalar("Validation_Train/Idempotence", avg_val_loss_idem, epoch + 1)
        writer.add_scalar("Validation_Train/Tightness", avg_val_loss_tight, epoch + 1)

        # In eval mode
        self.model.eval()
        self.model_copy.eval()
        avg_val_total_loss, avg_val_loss_rec, avg_val_loss_idem, avg_val_loss_tight = self._validate(data_loader, epoch)

        writer.add_scalar("Validation/Total", avg_val_total_loss, epoch + 1)
        writer.add_scalar("Validation/Reconstruction", avg_val_loss_rec, epoch + 1)
        writer.add_scalar("Validation/Idempotence", avg_val_loss_idem, epoch + 1)
        writer.add_scalar("Validation/Tightness", avg_val_loss_tight, epoch + 1)

        return (
            avg_val_total_loss,
            avg_val_loss_rec,
            avg_val_loss_idem,
            avg_val_loss_tight,
        )

    def log_scores(self, data_loader: DataLoader, writer: SummaryWriter, epoch: int):
        use_fourier_sampling = self.config["training"].get(
            "use_fourier_sampling",
            False,
        )
        num_images = 200

        self.model.eval()
        _, generated = rec_generate_images(
            model=self.model,
            device=self.device,
            data=data_loader,
            n_images=num_images,
            n_recursions=1,
            reconstruct=False,
            use_fourier_sampling=use_fourier_sampling,
        )
        self.model.train()
        _, generated_train = rec_generate_images(
            model=self.model,
            device=self.device,
            data=data_loader,
            n_images=num_images,
            n_recursions=1,
            reconstruct=False,
            use_fourier_sampling=use_fourier_sampling,
        )

        # Calculate FID/IS scores
        real_images = torch.zeros((num_images, *next(iter(data_loader))[0].shape[1:]))
        for i, (_x, _) in enumerate(data_loader):
            if (i+1)*_x.shape[0] < num_images:
                real_images[i*_x.shape[0]:(i+1)*_x.shape[0]] = _x
            else:
                remainder = num_images - i*_x.shape[0]
                real_images[i*_x.shape[0]:] = _x[:remainder]
                break

        normalized_generated = normalize_batch(generated[:, 0])
        normalized_generated_train = normalize_batch(generated_train[:, 0])
        normalized_real_images = normalize_batch(real_images)
        if real_images.shape[1] == 1:
            # If we have single channel images, repeat first channel 3 times
            normalized_real_images = normalized_real_images.repeat(1, 3, 1, 1)
            normalized_generated = normalized_generated.repeat(1, 3, 1, 1)
            normalized_generated_train = normalized_generated_train.repeat(1, 3, 1, 1)

        # Calculated on cpu due to limited GPU memory
        fid_score, inception_score, inception_deviation = evaluate_generator(generated_images=normalized_generated, real_images=normalized_real_images, batch_size=100, normalized_images=True, device="cpu")
        writer.add_scalar("Validation/FID", fid_score, epoch + 1)
        writer.add_scalar("Validation/IS", inception_score, epoch + 1)
        fid_score, inception_score, inception_deviation = evaluate_generator(generated_images=normalized_generated_train, real_images=normalized_real_images, batch_size=100, normalized_images=True, device="cpu")
        writer.add_scalar("Validation_Train/FID", fid_score, epoch + 1)
        writer.add_scalar("Validation_Train/IS", inception_score, epoch + 1)

    def log_images(
        self,
        data_loader: DataLoader | tqdm,
        n_images: int,
        n_recursions: int,
        writer: SummaryWriter,
        epoch: int,
    ):
        use_fourier_sampling = self.config["training"].get(
            "use_fourier_sampling",
            False,
        )

        self.model.eval()
        original, reconstructed = rec_generate_images(
            model=self.model,
            device=self.device,
            data=data_loader,
            n_images=n_images,
            n_recursions=n_recursions,
            reconstruct=True,
            use_fourier_sampling=use_fourier_sampling,
        )
        noise, generated = rec_generate_images(
            model=self.model,
            device=self.device,
            data=data_loader,
            n_images=n_images,
            n_recursions=n_recursions,
            reconstruct=False,
            use_fourier_sampling=use_fourier_sampling,
        )
        writer.add_images("Image/Generated", normalize_batch(generated[:n_images, 0].detach()), epoch + 1)
        writer.add_images("Image/Noise", normalize_batch(noise[:n_images].detach()), epoch + 1)
        writer.add_images("Image/Reconstructed", normalize_batch(reconstructed[:n_images, 0].detach()), epoch + 1)
        writer.add_images("Image/Original", normalize_batch(original[:n_images].detach()), epoch + 1)

        self.model.train()
        original, reconstructed = rec_generate_images(
            model=self.model,
            device=self.device,
            data=data_loader,
            n_images=n_images,
            n_recursions=n_recursions,
            reconstruct=True,
            use_fourier_sampling=use_fourier_sampling,
        )
        noise, generated = rec_generate_images(
            model=self.model,
            device=self.device,
            data=data_loader,
            n_images=n_images,
            n_recursions=n_recursions,
            reconstruct=False,
            use_fourier_sampling=use_fourier_sampling,
        )
        writer.add_images("Image_Train/Generated", normalize_batch(generated[:n_images, 0].detach()), epoch + 1)
        writer.add_images("Image_Train/Noise", normalize_batch(noise[:n_images].detach()), epoch + 1)
        writer.add_images("Image_Train/Reconstructed", normalize_batch(reconstructed[:n_images, 0].detach()), epoch + 1)
        writer.add_images("Image_Train/Original", normalize_batch(original[:n_images].detach()), epoch + 1)

    def train_one_step(self, loss):
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def train_one_epoch(
        self, data_loader: DataLoader | tqdm, writer: SummaryWriter, epoch: int
    ):
        lambda_rec = self.config["losses"]["lambda_rec"]
        lambda_idem = self.config["losses"]["lambda_idem"]
        lambda_tight_end = self.config["losses"]["lambda_tight"]
        tight_clamp = self.config["losses"]["tight_clamp"]
        tight_clamp_ratio = self.config["losses"]["tight_clamp_ratio"]
        use_amp = self.config["training"]["use_amp"]

        # Manifold Warmup Parameters
        warmup_config = self.config["training"].get("manifold_warmup", {})
        warmup_enabled = warmup_config.get("enabled", False)
        warmup_epochs = warmup_config.get("warmup_epochs", 0)
        lambda_tight_start = warmup_config.get("lambda_tight_start", lambda_tight_end)
        schedule_type = warmup_config.get("schedule_type", "linear")

        # Calculate current lambda_tight based on warmup schedule
        if warmup_enabled and epoch < warmup_epochs:
            if schedule_type == "linear":
                lambda_tight = lambda_tight_start + (
                    lambda_tight_end - lambda_tight_start
                ) * (epoch / warmup_epochs)
            elif schedule_type == "exponential":
                lambda_tight = lambda_tight_start * (
                    lambda_tight_end / lambda_tight_start
                ) ** (epoch / warmup_epochs)
            else:
                raise ValueError(f"Unsupported schedule_type: {schedule_type}")
        else:
            lambda_tight = lambda_tight_end

        avg_loss_total = 0.0
        avg_loss_rec = 0.0
        avg_loss_idem = 0.0
        avg_loss_tight = 0.0
        train_batches = 0

        self.model.train()
        self.model_copy.train()
        for x, _ in data_loader:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                x = x.to(self.device)

                total_loss, loss_rec, loss_idem, loss_tight = self.get_losses(
                    x=x,
                    lambda_rec=lambda_rec,
                    lambda_idem=lambda_idem,
                    lambda_tight=lambda_tight,
                    tight_clamp=tight_clamp,
                    tight_clamp_ratio=tight_clamp_ratio,
                )

            self.train_one_step(total_loss)

            avg_loss_total += total_loss.item()
            avg_loss_rec += loss_rec.item()
            avg_loss_idem += loss_idem.item()
            avg_loss_tight += loss_tight.item()
            train_batches += 1

        writer.add_scalar("Loss/Total", avg_loss_total / train_batches, epoch + 1)
        writer.add_scalar("Loss/Reconstruction", avg_loss_rec / train_batches, epoch + 1)
        writer.add_scalar("Loss/Idempotence", avg_loss_idem / train_batches, epoch + 1)
        writer.add_scalar("Loss/Tightness", avg_loss_tight / train_batches, epoch + 1)
        writer.add_scalar("Hyperparameters/Lambda_Tight", lambda_tight, epoch + 1)

    def fit(
        self,
        data_loader: DataLoader | tqdm,
        val_data_loader: DataLoader | tqdm,
        writer: SummaryWriter,
    ):
        n_epochs = self.config["training"]["n_epochs"]
        save_period = self.config["training"]["save_period"]
        image_log_period = self.config["training"].get("image_log_period", 100)
        validation_period = self.config["training"].get("validation_period", 1)
        score_log_period = self.config['training'].get('score_log_period', 5)
        run_id = self.config["run_id"]

        # Early Stopping Parameters
        patience = self.config["early_stopping"].get("patience", 5)
        best_val_loss = float("inf")
        epochs_without_improvement = 0
        initial_validation_threshold = self.config["training"]["initial_validation_threshold"]

        writer.add_text("config", f"``` {self.config} ```")

        for epoch in tqdm(
            range(self.config.get("start_epoch", 0), n_epochs),
            position=1,
            desc="Epoch",
            total=n_epochs,
            initial=self.config.get("start_epoch", 0),
        ):
            epoch_timer = time.time()
            self.config["current_epoch"] = epoch  # Used when terminating training

            data_loader: tqdm = tqdm(
                data_loader, total=len(data_loader), position=0, desc="Train Step"
            )
            self.train_one_epoch(data_loader=data_loader, epoch=epoch, writer=writer)

            if (epoch + 1) % validation_period == 0 or (epoch + 1) == n_epochs:
                # Validation after epoch
                (
                    avg_val_total_loss,
                    avg_val_loss_rec,
                    avg_val_loss_idem,
                    avg_val_loss_tight,
                ) = self.log_validation(val_data_loader, writer, epoch)
                data_loader.write(
                    f"Epoch [{epoch+1}/{n_epochs}], Validation Losses -> "
                    f"Total: {avg_val_total_loss:.4f}, "
                    f"Reconstruction: {avg_val_loss_rec:.4f}, "
                    f"Idempotence: {avg_val_loss_idem:.4f}, "
                    f"Tightness: {avg_val_loss_tight:.4f}"
                )

                # Early Stopping Check
                if avg_val_total_loss < best_val_loss:
                    if (best_val_loss == float("inf") and avg_val_total_loss >= initial_validation_threshold):
                        data_loader.write(
                            f"Validation loss, {avg_val_total_loss:.4f} / {initial_validation_threshold:.4f}, too high restarting!"
                        )
                        return False  # Did not meet initial validation threshold

                    best_val_loss = avg_val_total_loss
                    epochs_without_improvement = 0
                    # Save the best model
                    checkpoint_path = os.path.join(
                        self.config["checkpoint"]["save_dir"], f"{run_id}_best_model.pt"
                    )
                    self.save_model(epoch + 1, checkpoint_path)
                    data_loader.write(
                        f"Validation loss improved to {best_val_loss:.4f}, saved best model."
                    )
                else:
                    epochs_without_improvement += validation_period
                    data_loader.write(
                        f"Validation loss did not improve for {epochs_without_improvement} epochs, current best is {best_val_loss}."
                    )
                    if epochs_without_improvement >= patience:
                        data_loader.write(
                            f"Early stopping triggered after {patience} epochs without improvement."
                        )
                        break  # Break out of the training loop

            if (epoch + 1) % image_log_period == 0 or (epoch + 1) == n_epochs:
                # Save the sampled image
                self.log_images(
                    data_loader, n_images=5, n_recursions=1, writer=writer, epoch=epoch
                )
                data_loader.write(f"Logged images for epoch [{epoch+1}/{n_epochs}]")

            if (epoch + 1) % score_log_period == 0 or (epoch + 1) == n_epochs:
                self.log_scores(data_loader, writer, epoch)
                data_loader.write(f"Logged scores for epoch [{epoch+1}/{n_epochs}]")

            if (epoch + 1) % save_period == 0 or (epoch + 1) == n_epochs:
                checkpoint_path = os.path.join(
                    self.config["checkpoint"]["save_dir"],
                    f"{run_id}_epoch_{epoch+1}.pt",
                )
                self.save_model(epoch + 1, checkpoint_path)
                data_loader.write(
                    f"Saved checkpoint at epoch {epoch+1} \nEpoch [{epoch+1}/{n_epochs}]"
                )

            epoch_time = time.time() - epoch_timer
            writer.add_scalar("Logs/Epoch_Timer", epoch_time, epoch + 1)

        return True
