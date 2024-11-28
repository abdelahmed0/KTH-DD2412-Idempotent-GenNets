import pathlib
import numpy as np
import cv2
from tensorboard.backend.event_processing import event_accumulator


def extract_tensorboard_images(path, outdir, image_log_period=10):
    """
        path: path to tensorboard logs
        outdir: image output dir
        image_log_period: what the image log period was to correctly label images
    """
    event_acc = event_accumulator.EventAccumulator(
        path, size_guidance={'images': 0})
    event_acc.Reload()

    outdir = pathlib.Path(outdir)
    outdir.mkdir(exist_ok=True, parents=True)

    for tag in event_acc.Tags()['images']:
        events = event_acc.Images(tag)

        tag_name = tag.replace('/', '_')
        dirpath = outdir / tag_name
        dirpath.mkdir(exist_ok=True, parents=True)

        for index, event in enumerate(events):
            s = np.frombuffer(event.encoded_image_string, dtype=np.uint8)
            image = cv2.imdecode(s, cv2.IMREAD_COLOR)
            outpath = dirpath / 'epoch_{:04}.jpg'.format((index+1)*image_log_period)
            cv2.imwrite(outpath.as_posix(), image)

