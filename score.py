import torch
import argparse
import numpy as np

from torch.utils.data import DataLoader
from tqdm import tqdm

from util.dataset import load_mnist, load_celeb_a
from util.model_util import load_model, load_checkpoint
from util.function_util import fourier_sample, normalize_batch
from util.scoring import evaluate_generator


def generate_and_score(
    model, device, data: DataLoader, n_images, n_recursions, batch_size
):
    assert n_recursions > 0, f"n_recursions must be greater than 0! Got {n_recursions}"
    sampled_images = torch.tensor([])
    real_images = torch.tensor([])
    with torch.no_grad():
        for i, (x, _) in tqdm(
            enumerate(data),
            total=np.ceil(n_images / batch_size) if n_images else len(data),
        ):
            if n_images and sampled_images.shape[0] >= n_images:
                break
            x = x.to(device)

            if use_fourier_sampling:
                z = fourier_sample(x)
            else:
                z = torch.randn_like(x).to(device)

            for _ in range(n_recursions):
                z = model(z)

            sampled_images = torch.cat([sampled_images, z.cpu()], dim=0)
            real_images = torch.cat([real_images, x.cpu()], dim=0)

    normalized_sampled_images = normalize_batch(sampled_images)
    normalized_real_images = normalize_batch(real_images)

    if real_images.shape[1] == 1:
        # If we have single channel images, repeat first channel 3 times
        normalized_real_images = normalized_real_images.repeat(1, 3, 1, 1)
        normalized_sampled_images = normalized_sampled_images.repeat(1, 3, 1, 1)

    # Calculated on cpu due to limited GPU memory
    ### NOTE: Change batchsize here if you run out of gpu memory,
    ###       could also change device to "cpu" in case you dont have enough vram
    fid_score, inception_score, inception_deviation = evaluate_generator(
        generated_images=normalized_sampled_images,
        real_images=normalized_real_images,
        batch_size=200,
        normalized_images=True,
        device=device,
    )

    print(
        f"FID: {fid_score}, IS: {inception_score}, IS deviation: {inception_deviation}"
    )
    print(
        f"Sample sizes, sampled: {normalized_sampled_images.shape}, real: {normalized_real_images.shape}"
    )

    return fid_score, inception_score, inception_deviation


if __name__ == "__main__":
    """Usage: python score.py --run_id <run_id> --epoch <epoch>"""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--num_recursions", type=int, required=False, default=1)
    parser.add_argument("--eval", type=str, required=True)
    args = parser.parse_args()

    # Setup
    normalized = True # True if images are in range [-1, 1]

    # Loading model and data
    n_images = None # None for all
    n_recursions = args.num_recursions
    run_id = args.run_id

    checkpoint = load_checkpoint(run_id)
    model = load_model(checkpoint)

    use_mnist = checkpoint['config']['dataset']['name'].lower() == "mnist"
    use_fourier_sampling = checkpoint['config']['training']['use_fourier_sampling']
    batch_size = checkpoint['config']['training']['batch_size']

    if use_mnist:
        _, data = load_mnist(batch_size=batch_size, single_channel=True)
    else:
        data = load_celeb_a(batch_size=batch_size, split='test')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.eval_mode.lower() in ['true', 'yes', 'y']:
        model.eval()
    else:
        model.train()

    generate_and_score(model, device, data, n_images, n_recursions, batch_size)
