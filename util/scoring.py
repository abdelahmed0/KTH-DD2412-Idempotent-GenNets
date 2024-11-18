import typing
import torch
from tqdm import tqdm
from torchmetrics.image.fid import FrechetInceptionDistance # https://lightning.ai/docs/torchmetrics/stable/image/frechet_inception_distance.html
from torchmetrics.image.inception import InceptionScore # https://lightning.ai/docs/torchmetrics/stable/image/inception_score.html

import gc

def evaluate_generator(generated_images: torch.Tensor, real_images: torch.Tensor, batch_size: int, normalized_images=False, device=None, skip_fid=False) -> typing.Tuple[float, float]:
    """
    Computes and returns the FID and IS scores. First float is mean fid value, second is mean inception score 
    over subsets and last float is the standard deviation of inception score over subsets.
    Set normalized_images to true if images are normalized to 0-1, else false if images are in the range [0-255].
    """
    device = device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")

    # Check which feature to use
    fid_score = 0
    if not skip_fid:
        image_size = generated_images.shape[1:]   # shape without batch

        fid = FrechetInceptionDistance(input_img_size=image_size, normalize=normalized_images).set_dtype(torch.float64).to(device)

        for batch in tqdm(range(len(generated_images)//batch_size), "FID calculating score"):
            fid.update(imgs=real_images[batch*batch_size:(batch+1)*batch_size,:,:,:], real=True)
            fid.update(imgs=generated_images[batch*batch_size:(batch+1)*batch_size,:,:,:], real=False)

        fid_score = fid.compute().item()

        del fid
        gc.collect()
        torch.cuda.empty_cache()

    inception = InceptionScore(normalize=normalized_images).to(device)

    for batch in tqdm(range(len(generated_images)//batch_size), "IS calculating score"):
        inception.update(generated_images[batch*batch_size:(batch+1)*batch_size,:,:,:])
    
    inception_score, inception_deviation = inception.compute()

    return (fid_score, inception_score.item(), inception_deviation.item())