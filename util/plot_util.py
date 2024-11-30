import matplotlib.pyplot as plt

from util.function_util import normalize_batch

def _plot_images(original: list, generated: list, grayscale=True, transformed=False):
    n_images = len(original)
    n_recursions = len(generated[0])

    fig, axs = plt.subplots(1+n_recursions, n_images+1, figsize=(n_images, n_recursions + 1))
    for col in range(n_images):
        for row in range(n_recursions+1):
            if col == 0:
                # Display the correct label for each row
                if row == 0:
                    label = "$x$"
                elif transformed and row == 1:
                    label = "$g(x)$"
                elif row <= 3:
                    # Generate nested notation like f(f(...f(x)...))
                    label = "$" + "f(" * row + "x" + ")" * row + "$"
                else:
                    label = f"$f^{row}(x)$"
                axs[row, col].text(0.9, 0.5, label, ha="right", va="center", fontsize=12)
                axs[row, col].axis('off')

            if row == 0:
                if grayscale:
                    axs[0, col+1].imshow(original[col, 0], cmap='gray')
                else:
                    axs[0, col+1].imshow(original[col].permute((1,2,0))) # imshow expects [width, height, channels]
                axs[0, col+1].axis('off')
            else:
                if grayscale:
                    axs[row, col+1].imshow(generated[col, row-1, 0], cmap='gray')
                else:
                    axs[row, col+1].imshow(generated[col, row-1].permute((1,2,0)))
                axs[row, col+1].axis('off')

    return fig, axs

def plot_images(original: list, generated: list, grayscale=True, normalized=False, transformed=False):
    """
    original: [n_images]
    generated: [n_images, n_recursions]
    """
    if normalized:
        original = normalize_batch(original)
        generated = normalize_batch(generated)

    fig, axs = _plot_images(original, generated, grayscale, transformed)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()


def save_images(original, generated, grayscale=False, normalized=True, output_path="output_images.png", suptitle=None, transformed=False):
    """
    Save the original and generated images to a file.
    
    Parameters:
        original (torch.Tensor): The original images of shape (n_images, C, H, W).
        generated (torch.Tensor): The generated images of shape (n_images, n_recursions, C, H, W).
        grayscale (bool): Whether the images are grayscale (single-channel).
        normalized (bool): Whether the images are normalized to [-1, 1].
        output_path (str): Filepath to save the resulting image.
    """    
    # Denormalize images if normalized
    if normalized:
        original = normalize_batch(original)
        generated = normalize_batch(generated)
    
    fig, axs = _plot_images(original, generated, grayscale, transformed)

    if suptitle:
        plt.suptitle(suptitle, fontsize=10, wrap=True)

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.savefig(output_path)
    plt.close(fig)