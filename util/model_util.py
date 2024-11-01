import torch

from model.dcgan import DCGAN

def load_model(path) -> DCGAN:
    checkpoint = torch.load(path)

    architecture = checkpoint['config']['model']['architecture']
    model = DCGAN(architecture=architecture)

    model.load_state_dict(checkpoint['model_state_dict'])
    return model