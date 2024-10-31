import torch
import argparse
from torch.utils.data import DataLoader
from tqdm import tqdm

from dcgan import DCGAN
from util.dataset import load_mnist
from util.scoring import evaluate_generator


def load_model(path):
    model = DCGAN()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def generate_and_score(model, device, mnist: DataLoader, n_batches, n_recursions):
    model.eval()

    sampled_images = torch.tensor([])
    with torch.no_grad():
        for i, (x, _) in tqdm(enumerate(mnist), total=n_batches):
            if i == n_batches:
                break
            x_hat = x.to(device)

            for _ in range(n_recursions):
                x_hat = model(x_hat)

            sampled_images = torch.cat([sampled_images, x_hat.cpu()], dim=0)
    

    real_images = mnist.dataset.data
    fid_score, is_score, is_deviation = evaluate_generator(generated_images=sampled_images.to(device), real_images=real_images, batch_size=128, normalized_images=True, skip_fid=True, device=device)
    print(f'FID: {fid_score}, IS: {is_score}, IS deviation: {is_deviation}')
    print(f'Sample sizes, sampled: {sampled_images.shape}, real: {real_images.shape}')


if __name__=="__main__":
    """Usage: python score.py --run_id <run_id> --epoch <epoch>""" 
    ## TODO: This should be used on some more complex dataset, mnist being single channel makes it harder to compute and kind of unecessary... 
    ## TODO: Still under dev, we need to generate new images from noise, not reconstructions

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--epoch", type=int, required=True)
    args = parser.parse_args()

    # Setup
    n_batches = 120
    batch_size = 512
    n_recursions = 1
    run_id = args.run_id
    epoch = args.epoch

    model = load_model(f"checkpoints/{run_id}_{epoch}.pt")
    mnist = load_mnist(batch_size=batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    generate_and_score(model, device, mnist, n_batches, n_recursions)

