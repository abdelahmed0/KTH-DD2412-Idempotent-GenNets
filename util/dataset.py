import torch
from typing import Tuple
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.transforms import Lambda
from torchvision.datasets.mnist import MNIST
from torchvision.datasets.celeba import CelebA


def add_noise_(x):
    return (x + torch.randn_like(x) * 0.15).clamp(-1, 1)

def load_mnist(
    batch_size=256,
    download=True,
    num_workers=3,
    pin_memory=False,
    single_channel=False,
    path="./data",
    validation_split=0.1,
    add_noise=False,
) -> Tuple[DataLoader, DataLoader]:
    if single_channel:
        trans = [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
        if add_noise:
            trans.append(Lambda(add_noise_))

        transform = transforms.Compose(trans)
    else:
        transform = transforms.Compose(
            [
                transforms.Resize(64),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
                transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalize to [-1, 1]
            ]
        )

    dataset = MNIST(root=path, download=download, transform=transform)

    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return train_loader, val_loader


def load_celeb_a(
    batch_size=256,
    download=True,
    num_workers=3,
    pin_memory=False,
    random_flip=False,
    split="train",
    path="./data",
) -> DataLoader:
    transform = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.CenterCrop(64),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),  # Normalize to [-1, 1]
        ]
        +
        ([
            transforms.RandomHorizontalFlip(p=0.5)
        ] if random_flip else [])
    )

    celebA = DataLoader(
        CelebA(root=path, split=split, transform=transform, download=download),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return celebA

if __name__ == '__main__':
    mnist_3 = load_mnist(batch_size=32, single_channel=False)[0]
    mnist_3_image = next(iter(mnist_3))[0]
    assert mnist_3_image.shape == (32, 3, 64, 64)
    assert mnist_3_image.min() >= -1
    assert mnist_3_image.max() <= 1
    assert mnist_3_image.min() < 0
    assert mnist_3_image.max() > 0

    mnist = load_mnist(batch_size=32, single_channel=True)[0]
    mnist_image = next(iter(mnist))[0]
    assert mnist_image.shape == (32, 1, 28, 28)
    assert mnist_image.min() >= -1
    assert mnist_image.max() <= 1
    assert mnist_image.min() < 0
    assert mnist_image.max() > 0

    celeb = load_celeb_a(batch_size=32)
    celeb_image = next(iter(celeb))[0]
    assert celeb_image.shape == (32, 3, 64, 64), f"Got {celeb_image.shape}, not (32, 3, 64, 64)"
    assert celeb_image.min() >= -1
    assert celeb_image.max() <= 1
    assert celeb_image.min() < 0
    assert celeb_image.max() > 0

    print("Passed all tests! Wow, good job you mediocre programmer")