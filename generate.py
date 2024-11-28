import torch
import argparse

from util.dataset import load_mnist, load_celeb_a
from util.plot_images import plot_images
from util.model_util import load_model, load_checkpoint
from util.function_util import fourier_sample

def rec_generate_images(model, device, data, n_images, n_recursions, reconstruct, use_fourier_sampling, with_label=False):
    assert n_recursions > 0, f"n_recursions must be greater than 0! Got {n_recursions}"
    original = torch.empty(n_images, *next(iter(data))[0].shape[1:]) # []
    reconstructed = torch.empty(n_images, n_recursions, *next(iter(data))[0].shape[1:]) #[[] for _ in range(n_images)]
    with torch.no_grad():
        if reconstruct:
            for i, (x, y) in enumerate(data):
                if i == n_images:
                    break

                original[i] = x[0]
                x_hat = x.to(device)
                y = y.to(device) #, dtype=torch.float)

                for j in range(n_recursions):
                    if with_label:
                        x_hat = model(x_hat, y)
                    else:
                        x_hat = model(x_hat)
                    reconstructed[i, j] = x_hat[0].cpu()
        else:
            batch, y = next(iter(data))
            y = y.to(device) #, dtype=torch.float)
            for i in range(n_images):
                if use_fourier_sampling:
                    x = fourier_sample(batch)
                else:
                    x = torch.randn_like(batch).to(device)
                original[i] = x[0].clamp(-1.0, 1.0).cpu()
                x_hat = x.to(device)
                
                rand_i = torch.randint(0, y.shape[0], (1,))
                y_hat = y[rand_i]
                for j in range(n_recursions):
                    if with_label:
                        x_hat = model(x_hat, y_hat)
                    else:
                        x_hat = model(x_hat)
                    reconstructed[i, j] = x_hat[0].cpu()
    
    return original, reconstructed
    

if __name__=="__main__":
    """Usage: python generate.py --run_id <run_id>""" 
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    args = parser.parse_args()
    
    # Setup
    n_images = 10
    n_recursions = 3
    normalized = True # True if images are in range [-1, 1]

    # Loading model and data
    run_id = args.run_id

    checkpoint = load_checkpoint(f"checkpoints/{run_id}.pt")
    model = load_model(checkpoint)

    use_mnist = checkpoint['config']['dataset']['name'].lower() == "mnist"
    use_fourier_sampling = checkpoint['config']['training']['use_fourier_sampling']
    grayscale = True if use_mnist else False
    
    if use_mnist:
        _, data = load_mnist(batch_size=checkpoint['config']['training']['batch_size'], single_channel=True)
    else:
        data = load_celeb_a(batch_size=checkpoint['config']['training']['batch_size'], split='test')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    #model.eval()
    
    original, reconstructed = rec_generate_images(model=model, device=device, data=data, n_images=n_images, n_recursions=n_recursions, reconstruct=True, use_fourier_sampling=use_fourier_sampling)
    plot_images(original, reconstructed, grayscale=grayscale, normalized=normalized)
    original, reconstructed = rec_generate_images(model=model, device=device, data=data, n_images=n_images, n_recursions=n_recursions, reconstruct=False, use_fourier_sampling=use_fourier_sampling)
    plot_images(original, reconstructed, grayscale=grayscale, normalized=normalized)

