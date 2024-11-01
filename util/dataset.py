from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.celeba import CelebA


def load_mnist(
    batch_size=256,
    download=True,
    num_workers=3,
    pin_memory=False,
    single_channel=False,
    path="./data",
) -> DataLoader:
    if single_channel:
        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),  # Normalize to [-1, 1]
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalize to [-1, 1]
            ]
        )

    mnist = DataLoader(
        MNIST(root=path, download=download, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return mnist


def load_celeb_a(
    batch_size=256,
    download=True,
    num_workers=3,
    pin_memory=False,
    split="train",
    path="./data",
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.CenterCrop(64),
            transforms.Normalize([0.5]* 3, [0.5] * 3),  # Normalize to [-1, 1]
        ]
    )

    celebA = DataLoader(
        CelebA(root=path, split=split, transform=transform, download=download),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return celebA

if __name__ == '__main__':
    mnist_3 = load_mnist(batch_size=32, single_channel=False)
    mnist_3_image = next(iter(mnist_3))[0]
    assert mnist_3_image.shape == (32, 3, 64, 64)
    assert mnist_3_image.min() >= -1
    assert mnist_3_image.max() <= 1
    assert mnist_3_image.min() < 0
    assert mnist_3_image.max() > 0

    mnist = load_mnist(batch_size=32, single_channel=True)
    mnist_image = next(iter(mnist))[0]
    assert mnist_image.shape == (32, 1, 28, 28)
    assert mnist_image.min() >= -1
    assert mnist_image.max() <= 1
    assert mnist_image.min() < 0
    assert mnist_image.max() > 0

    celeb = load_celeb_a(batch_size=32)
    celeb_image = next(iter(celeb))[0]
    assert celeb_image.shape == (32, 3, 64, 64)
    assert celeb_image.min() >= -1
    assert celeb_image.max() <= 1
    assert celeb_image.min() < 0
    assert celeb_image.max() > 0

    print("Passed all tests! Wow, good job you mediocre programmer")