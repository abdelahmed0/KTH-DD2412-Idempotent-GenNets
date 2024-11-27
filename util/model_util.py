import torch

from model.dcgan import DCGAN


def load_checkpoint(path):
    checkpoint = torch.load(path)

    return checkpoint


def load_model(checkpoint) -> DCGAN:
    architecture = checkpoint["config"]["model"]["architecture"]
    model = DCGAN(
        architecture=architecture,
        norm=checkpoint["config"]["model"].get("norm", "batchnorm"),
        use_bias=checkpoint["config"]["model"].get("use_bias", False),
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    return model
