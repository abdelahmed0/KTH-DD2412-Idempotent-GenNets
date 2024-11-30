import torch

from model.dcgan import DCGAN
from model.u_net import UNet
from model.u_net_conditional import UNetConditional


def load_checkpoint(path):
    checkpoint = torch.load(path)

    return checkpoint


def load_model(checkpoint, device="cpu", force_conditional=False) -> DCGAN:
    # Force_conditional due to saved config being wrong 
    architecture = checkpoint["config"]["model"]["architecture"]

    if "dcgan" in architecture.lower():
        model = DCGAN(
            architecture=architecture,
            norm=checkpoint["config"]["model"].get("norm", "batchnorm"),
            use_bias=checkpoint["config"]["model"].get("use_bias", True),
        )
    elif "unet" in architecture.lower().replace("_",""):
        if "conditional" in architecture.lower() or force_conditional:
            model = UNetConditional(device)
        else:
            model = UNet()
    

    model.load_state_dict(checkpoint["model_state_dict"])
    return model
