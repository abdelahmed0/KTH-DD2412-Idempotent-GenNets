from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets.mnist import MNIST


def load_mnist(
    batch_size=256,
    download=True,
    num_workers=3,
    pin_memory=False,
    single_channel=False,
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
                transforms.Normalize([0.5 * 3], [0.5] * 3),  # Normalize to [-1, 1]
            ]
        )

    mnist = DataLoader(
        MNIST(root="./data", download=download, transform=transform),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return mnist
