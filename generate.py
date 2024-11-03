import torch
import argparse

from model.dcgan import DCGAN
from util.dataset import load_mnist, load_celeb_a
from util.plot_images import plot_images
from util.model_util import load_model

def rec_generate_images(model, device, mnist, image_shape, n_images, n_recursions, reconstruct, grayscale):
    model.eval()
    original = []
    reconstructed = [[] for _ in range(n_images)]
    with torch.no_grad():
        if reconstruct:
            for i, (x, _) in enumerate(mnist):
                if i == n_images:
                    break
                original.append((x.numpy() * 0.5) + 0.5) # from pixel values [-1,1] to [0,1]
                x_hat = x.to(device)

                for _ in range(n_recursions):
                    x_hat = model(x_hat)
                    reconstructed[i].append((x_hat.cpu().numpy() * 0.5) + 0.5)
        else:
            for i in range(n_images):
                x = torch.randn(1, image_shape[0], image_shape[1], image_shape[2]).to(device)
                original.append(x.cpu().numpy())
                x_hat = x.to(device)

                for _ in range(n_recursions):
                    x_hat = model(x_hat)
                    reconstructed[i].append((x_hat.cpu().numpy() * 0.5) + 0.5)
        
    plot_images(original, reconstructed, grayscale=grayscale)

if __name__=="__main__":
    """Usage: python generate.py --run_id <run_id> --epoch <epoch>"""    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()

    # Setup
    n_images = 10
    n_recursions = 3
    image_shape = (3,64,64)
    grayscale = False
    run_id = args.run_id
    epoch = args.epoch

    model = load_model(f"checkpoints/{run_id}_epoch_{epoch}.pt")
    mnist = load_celeb_a(batch_size=1, split="train")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    rec_generate_images(model=model, device=device, mnist=mnist, n_images=n_images, n_recursions=n_recursions, image_shape=image_shape, reconstruct=True, grayscale=grayscale)
    rec_generate_images(model=model, device=device, mnist=mnist, n_images=n_images, n_recursions=n_recursions, image_shape=image_shape, reconstruct=False, grayscale=grayscale)

