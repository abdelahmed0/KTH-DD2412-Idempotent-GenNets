import torch
import matplotlib.pyplot as plt

from dcgan import DCGAN
from dataset import load_mnist


def load_model(path):
    model = DCGAN()
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def reconstruct_mnist_images(model, device, mnist, n_images=10):
    model.eval()
    original = []
    reconstructed = []
    with torch.no_grad():
        for i, (x, _) in enumerate(mnist):
            if i == n_images:
                break
            x = x.to(device)
            x_hat = model(x)
            x = x.cpu().numpy()
            x_hat = x_hat.cpu().numpy()
            original.append(x)
            reconstructed.append(x_hat)
        
    fig, axs = plt.subplots(2, n_images, figsize=(20, 5))
    for i in range(n_images):
        axs[0, i].imshow(original[i][0, 0], cmap='gray')
        axs[1, i].imshow(reconstructed[i][0, 0], cmap='gray')
        axs[0, i].axis('off')
        axs[1, i].axis('off')
    plt.show()

if __name__=="__main__":
    run_id = "baseline_gaussian_latents"
    epoch = 10
    model = load_model(f"checkpoints/{run_id}_{epoch}.pt")
    mnist = load_mnist()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    reconstruct_mnist_images(model, device, mnist)

