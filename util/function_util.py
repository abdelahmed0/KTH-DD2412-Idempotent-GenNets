import torch

def fourier_sample(batch):
    f = torch.fft.fft2(batch) # 2D Fourier Transform
    # Real and Imaginary mean and std
    means = f.mean(dim=0)
    stds = f.std(dim=0)
    z = torch.fft.ifft2(torch.randn_like(f) * stds + means).real
    return z

def normalize_batch(batch: torch.Tensor):
    # [-1, 1] -> [0, 1]
    return ((batch * 0.5) + 0.5).clip(0.0, 1.0)

if __name__ == "__main__":
    import os
    from dataset import load_mnist
    from matplotlib import pyplot as plt
    from torch.utils.data import DataLoader
    from torchvision import transforms

    # CelebA fourier sampling
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(64),
            transforms.Normalize([0.5]* 3, [0.5] * 3),  # Normalize to [-1, 1]
        ]
    )

    images = []
    for i, file in enumerate(os.listdir('data/celeba')):
        if i == 10:
            break
        img = plt.imread(f'data/celeba/{file}')
        images.append(transform(img))

    celebA = DataLoader(images, batch_size=10, shuffle=False)

    batch = next(iter(celebA))
    print(batch.shape)
    z = fourier_sample(batch)

    fig, ax = plt.subplots(2, 10, figsize=(20, 2))
    for i in range(10):
        ax[0][i].imshow(batch[i].permute(1, 2, 0))
        ax[1][i].imshow(z[i][0])
        ax[0][i].axis('off')
        ax[1][i].axis('off')
    plt.show()

    # MNIST fourier sampling
    mnist = load_mnist()
    batch, _ = next(iter(mnist))
    z = fourier_sample(batch)

    fig, ax = plt.subplots(1, 10, figsize=(20, 2))
    for i in range(10):
        ax[i].imshow(z[i][0], cmap='gray')
        ax[i].axis('off')
    plt.show()
