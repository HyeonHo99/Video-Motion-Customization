import random
import torch


def short_size_scale(images, size):
    h, w = images.shape[-2:]
    short, long = (h, w) if h < w else (w, h)

    scale = size / short
    long_target = int(scale * long)

    target_size = (size, long_target) if h < w else (long_target, size)

    return torch.nn.functional.interpolate(
        input=images, size=target_size, mode="bilinear", antialias=True
    )


def random_short_side_scale(images, size_min, size_max):
    size = random.randint(size_min, size_max)
    return short_size_scale(images, size)


def random_crop(images, height, width):
    image_h, image_w = images.shape[-2:]
    h_start = random.randint(0, image_h - height)
    w_start = random.randint(0, image_w - width)
    return images[:, :, h_start : h_start + height, w_start : w_start + width]


def center_crop(images, height, width):
    image_h, image_w = images.shape[-2:]
    h_start = (image_h - height) // 2
    w_start = (image_w - width) // 2
    return images[:, :, h_start : h_start + height, w_start : w_start + width]