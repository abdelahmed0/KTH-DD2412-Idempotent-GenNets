import torch

from torch.nn import functional as F
from torch.utils.tensorboard.writer import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from trainer.ign_trainer import IGNTrainer
from util.function_util import fourier_sample, normalize_batch
from generate import rec_generate_images
from util.scoring import evaluate_generator


class IGNConditionalTrainer(IGNTrainer):
    def get_losses(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
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

        if torch.rand(1) < 0.1: # No label 10% of the time
            y = None #torch.empty(0, device=self.device, dtype=y.dtype)

        # x, z are float32
        # f_z, fz are float16
        self.model_copy.load_state_dict(self.model.state_dict())
        fx = self.model(x, y)
        fz = self.model(z, y)
        f_z = fz.detach()
        ff_z = self.model(f_z, y)
        f_fz = self.model_copy(fz, y)

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
            for x, y in data_loader:
                torch.compiler.cudagraph_mark_step_begin()
                x = x.to(self.device)
                y = y.to(self.device)

                total_loss, loss_rec, loss_idem, loss_tight = self.get_losses(
                    x=x,
                    y=y,
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
            with_label=True,
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
            with_label=True,
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
            with_label=True,
        )
        noise, generated = rec_generate_images(
            model=self.model,
            device=self.device,
            data=data_loader,
            n_images=n_images,
            n_recursions=n_recursions,
            reconstruct=False,
            use_fourier_sampling=use_fourier_sampling,
            with_label=True,
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
            with_label=True,
        )
        noise, generated = rec_generate_images(
            model=self.model,
            device=self.device,
            data=data_loader,
            n_images=n_images,
            n_recursions=n_recursions,
            reconstruct=False,
            use_fourier_sampling=use_fourier_sampling,
            with_label=True,
        )
        writer.add_images("Image_Train/Generated", normalize_batch(generated[:n_images, 0].detach()), epoch + 1)
        writer.add_images("Image_Train/Noise", normalize_batch(noise[:n_images].detach()), epoch + 1)
        writer.add_images("Image_Train/Reconstructed", normalize_batch(reconstructed[:n_images, 0].detach()), epoch + 1)
        writer.add_images("Image_Train/Original", normalize_batch(original[:n_images].detach()), epoch + 1)

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
        train_steps = 0

        self.model.train()
        self.model_copy.train()
        for x, y in data_loader:
            torch.compiler.cudagraph_mark_step_begin()
            with torch.autocast(
                device_type=self.device.type,
                dtype=torch.float16,
                enabled=use_amp,
            ):
                x = x.to(self.device)
                y = y.to(self.device)

                total_loss, loss_rec, loss_idem, loss_tight = self.get_losses(
                    x=x,
                    y=y,
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
            train_steps += 1

        writer.add_scalar("Loss/Total", avg_loss_total / train_steps, epoch + 1)
        writer.add_scalar("Loss/Reconstruction", avg_loss_rec / train_steps, epoch + 1)
        writer.add_scalar("Loss/Idempotence", avg_loss_idem / train_steps, epoch + 1)
        writer.add_scalar("Loss/Tightness", avg_loss_tight / train_steps, epoch + 1)
        writer.add_scalar("Hyperparameters/Lambda_Tight", lambda_tight, epoch + 1)