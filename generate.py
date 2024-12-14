import os
import torch
import argparse

from tqdm import tqdm

from util.dataset import load_mnist, load_celeb_a
from util.plot_util import plot_images, save_images
from util.model_util import load_model, load_checkpoint
from util.function_util import fourier_sample


def rec_generate_images(
    model,
    device,
    data,
    n_images,
    n_recursions,
    reconstruct,
    use_fourier_sampling,
    with_label=False,
    transforms: list = None,
    loading_bar=False,
    y_input_is_None=False,
):
    assert n_recursions > 0, f"n_recursions must be greater than 0! Got {n_recursions}"
    if transforms and not reconstruct:
        raise NotImplementedError("Transforms on generation is not supported yet!")

    original = torch.empty(n_images, *next(iter(data))[0].shape[1:])  # []
    reconstructed = torch.empty(
        n_images, n_recursions + (1 if transforms else 0), *next(iter(data))[0].shape[1:]
    )  # [[] for _ in range(n_images)]
    with torch.inference_mode():
        if reconstruct:
            for i, (x, y) in (
                enumerate(tqdm(data, total=n_images)) if loading_bar else enumerate(data)
            ):
                if i == n_images:
                    break

                original[i] = x[0].clamp(-1.0, 1.0).cpu()

                if transforms:
                    for transform in transforms:
                        x = transform(x)
                    reconstructed[i, 0] = x[0].clamp(-1.0, 1.0).cpu()

                x_hat = x.to(device)
                y = y.to(device)  # , dtype=torch.float)

                for j in range(n_recursions):
                    if with_label:
                        x_hat = model(x_hat, None if y_input_is_None else y)
                    else:
                        x_hat = model(x_hat)
                    reconstructed[i, j + (1 if transforms else 0)] = x_hat[0].cpu()
        else:
            batch, y = next(iter(data))
            y = y.to(device)  # , dtype=torch.float)
            batch = batch.to(device)
            for i in tqdm(range(n_images)) if loading_bar else range(n_images):
                if use_fourier_sampling:
                    x = fourier_sample(batch)
                else:
                    x = torch.randn_like(batch)

                rand_i = torch.randint(0, y.shape[0], (1,))
                x_hat = x
                y_hat = y[rand_i]

                original[i] = x_hat[rand_i].clamp(-1.0, 1.0).cpu()
                for j in range(n_recursions):
                    if with_label:
                        x_hat = model(x_hat, None if y_input_is_None else y_hat)
                    else:
                        x_hat = model(x_hat)
                    reconstructed[i, j] = x_hat[rand_i].cpu()

    return original, reconstructed


if __name__ == "__main__":
    """Usage: python generate.py --run_id <run_id>"""
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--eval_mode", type=str, required=True)
    args = parser.parse_args()

    # Setup
    num_samples = 3  # how many times to generate images
    n_images = 10
    n_recursions = 3
    normalized = True  # True if images are in range [-1, 1]
    
    os.makedirs("generated_images", exist_ok=True)

    # Loading model and data
    run_id = args.run_id

    checkpoint = load_checkpoint(run_id)
    model = load_model(checkpoint)

    use_mnist = checkpoint["config"]["dataset"]["name"].lower() == "mnist"
    use_fourier_sampling = checkpoint["config"]["training"]["use_fourier_sampling"]
    grayscale = True if use_mnist else False

    if use_mnist:
        _, data = load_mnist(
            batch_size=checkpoint["config"]["training"]["batch_size"],
            single_channel=True,
        )
    else:
        data = load_celeb_a(
            batch_size=checkpoint["config"]["training"]["batch_size"], split="test"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    if args.eval_mode.lower() in ['true', 'yes', 'y']:
        model.eval()
    else:
        model.train()

    for i in range(num_samples):
        original, reconstructed = rec_generate_images(
            model=model,
            device=device,
            data=data,
            n_images=n_images,
            n_recursions=n_recursions,
            reconstruct=True,
            use_fourier_sampling=use_fourier_sampling,
            loading_bar=True,
        )
        #plot_images(original, reconstructed, grayscale=grayscale, normalized=normalized)
        save_images(original, reconstructed, grayscale, normalized, output_path=f"generated_images/{run_id.split("/")[-1].removesuffix('.pt')}_reconstructed_{i}.png")
        original, reconstructed = rec_generate_images(
            model=model,
            device=device,
            data=data,
            n_images=n_images,
            n_recursions=n_recursions,
            reconstruct=False,
            use_fourier_sampling=use_fourier_sampling,
            loading_bar=True,
        )
        #plot_images(original, reconstructed, grayscale=grayscale, normalized=normalized)
        save_images(original, reconstructed, grayscale, normalized, output_path=f"generated_images/{run_id.split("/")[-1].removesuffix('.pt')}_generated_{i}.png")
