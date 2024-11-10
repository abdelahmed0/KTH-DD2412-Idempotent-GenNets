import torch
import argparse

from model.dcgan import DCGAN
from util.dataset import load_mnist, load_celeb_a
from util.plot_images import plot_images
from util.model_util import load_model
from util.function_util import fourier_sample

def rec_generate_images(model, device, data, n_images, n_recursions, reconstruct, use_fourier_sampling):
    model.eval()
    original = []
    reconstructed = [[] for _ in range(n_images)]
    with torch.no_grad():
        if reconstruct:
            for i, (x, _) in enumerate(data):
                if i == n_images:
                    break

                original.append(x) # from pixel values [-1,1] to [0,1]
                x_hat = x.to(device)

                for _ in range(n_recursions):
                    x_hat = model(x_hat)
                    reconstructed[i].append(x_hat.cpu())
        else:
            batch, _ = next(iter(data))
            for i in range(n_images):
                if use_fourier_sampling:
                    x = fourier_sample(batch)
                else:
                    x = torch.randn_like(batch).to(device)
                original.append(x.cpu())
                x_hat = x.to(device)

                for _ in range(n_recursions):
                    x_hat = model(x_hat)
                    reconstructed[i].append(x_hat.cpu())
    
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
    use_fourier_sampling = True
    n_images = 10
    n_recursions = 3

    
    grayscale = True if use_mnist else False
    run_id = args.run_id
    epoch = args.epoch

    model = load_model(f"checkpoints/{run_id}_epoch_{epoch}.pt")
    if use_mnist:
        data = load_mnist(batch_size=10, single_channel=True)
    else:
        data = load_celeb_a(batch_size=1, split='train')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    original, reconstructed = rec_generate_images(model=model, device=device, data=data, n_images=n_images, n_recursions=n_recursions, reconstruct=True, use_fourier_sampling=use_fourier_sampling)
    plot_images(original, reconstructed, grayscale=grayscale)
    original, reconstructed = rec_generate_images(model=model, device=device, data=data, n_images=n_images, n_recursions=n_recursions, reconstruct=False, use_fourier_sampling=use_fourier_sampling)
    plot_images(original, reconstructed, grayscale=grayscale)

