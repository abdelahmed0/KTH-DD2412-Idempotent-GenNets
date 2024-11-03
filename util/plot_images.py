import matplotlib.pyplot as plt


def plot_images(original: list, generated: list, grayscale=True):
    """
    original: [n_images]
    generated: [n_images, n_recursions]
    """
    n_images = len(original)
    n_recursions = len(generated[0])

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
                if grayscale:
                    axs[0, col+1].imshow(original[col][0, 0], cmap='gray')
                else:
                    axs[0, col+1].imshow(original[col][0].transpose((1,2,0))) # imshow expects [width, height, channels]
                axs[0, col+1].axis('off')
            else:
                if grayscale:
                    axs[row, col+1].imshow(generated[col][row-1][0, 0], cmap='gray')
                else:
                    axs[row, col+1].imshow(generated[col][row-1][0].transpose((1,2,0)))
                axs[row, col+1].axis('off')

    plt.subplots_adjust(wspace=0, hspace=0)
    plt.show()