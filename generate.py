import torch
import matplotlib.pyplot as plt
import argparse

from dcgan import DCGAN
from util.dataset import load_mnist


def load_model(path):
    model = DCGAN()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def reconstruct_mnist_images(model, device, mnist, n_images, n_recursions):
    model.eval()
    original = []
    reconstructed = [[] for _ in range(n_images)]
    with torch.no_grad():
        for i, (x, _) in enumerate(mnist):
            if i == n_images:
                break
            original.append(x.numpy())
            x_hat = x.to(device)

            for _ in range(n_recursions):
                x_hat = model(x_hat)
                reconstructed[i].append(x_hat.cpu().numpy())
        
    fig, axs = plt.subplots(1+n_recursions, n_images+1, figsize=(n_images, n_recursions + 1))
    for col in range(n_images):
        for row in range(n_recursions+1):
            if col == 0:
                # Display the correct label for each row
                if row == 0:
                    label = "$x$"
                elif row <= 3:
                    # Generate nested notation like f(f(...f(x)...))
                    label = "$" + "f(" * row + "x" + ")" * row + "$"
                else:
                    label = f"$f^{row}(x)$"
                axs[row, col].text(0.9, 0.5, label, ha="right", va="center", fontsize=12)
                axs[row, col].axis('off')

            if row == 0:
                axs[0, col+1].imshow(original[col][0, 0], cmap='gray')
                axs[0, col+1].axis('off')
            else:
                axs[row, col+1].imshow(reconstructed[col][row-1][0, 0], cmap='gray')
                axs[row, col+1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()

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
    run_id = args.run_id
    epoch = args.epoch

    model = load_model(f"checkpoints/{run_id}_epoch_{epoch}.pt")
    mnist = load_mnist(batch_size=1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    reconstruct_mnist_images(model, device, mnist, n_images, n_recursions)

