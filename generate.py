import torch
import argparse

from model.dcgan import DCGAN
from util.dataset import load_mnist, load_celeb_a
from util.plot_images import plot_images
from util.model_util import load_model
from util.function_util import fourier_sample

def rec_generate_images(model, device, data, n_images, n_recursions, reconstruct, use_fourier_sampling):
    model.eval()
    original = torch.empty(n_images, *next(iter(data))[0].shape[1:]) # []
    reconstructed = torch.empty(n_images, n_recursions, *next(iter(data))[0].shape[1:]) #[[] for _ in range(n_images)]
    with torch.no_grad():
        if reconstruct:
            for i, (x, _) in enumerate(data):
                if i == n_images:
                    break

                original[i] = x[0]
                x_hat = x.to(device)

                for j in range(n_recursions):
                    x_hat = model(x_hat)
                    reconstructed[i, j] = x_hat[0].cpu()
        else:
            batch, _ = next(iter(data))
            for i in range(n_images):
                if use_fourier_sampling:
                    x = fourier_sample(batch)
                else:
                    x = torch.randn_like(batch).to(device)
                original[i] = x[0].clamp(-1.0, 1.0).cpu()
                x_hat = x.to(device)

                for j in range(n_recursions):
                    x_hat = model(x_hat)
                    reconstructed[i, j] = x_hat[0].cpu()
    
    return original, reconstructed
    

if __name__=="__main__":
    """Usage: python generate.py --run_id <run_id> --epoch <epoch>""" 
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()


    # Setup
    use_mnist = True
    use_fourier_sampling = False
    n_images = 10
    n_recursions = 3
    normalized = True # True if images are in range [-1, 1]

    
    grayscale = True if use_mnist else False
    run_id = args.run_id
    epoch = args.epoch

    model = load_model(f"checkpoints/{run_id}_epoch_{epoch}.pt")
    if use_mnist:
        data, _ = load_mnist(batch_size=512 if use_fourier_sampling else 1, single_channel=True)
    else:
        data = load_celeb_a(batch_size=256 if use_fourier_sampling else 1, split='train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    original, reconstructed = rec_generate_images(model=model, device=device, data=data, n_images=n_images, n_recursions=n_recursions, reconstruct=True, use_fourier_sampling=use_fourier_sampling)
    plot_images(original, reconstructed, grayscale=grayscale, normalized=normalized)
    original, reconstructed = rec_generate_images(model=model, device=device, data=data, n_images=n_images, n_recursions=n_recursions, reconstruct=False, use_fourier_sampling=use_fourier_sampling)
    plot_images(original, reconstructed, grayscale=grayscale, normalized=normalized)

