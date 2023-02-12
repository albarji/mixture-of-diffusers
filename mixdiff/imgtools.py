import numpy as np
import torch
from PIL import Image, ImageFilter


def preprocess_image(image):
    """Preprocess an input image
    
    Same as https://github.com/huggingface/diffusers/blob/1138d63b519e37f0ce04e027b9f4a3261d27c628/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion_img2img.py#L44
    """
    w, h = image.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    image = image.resize((w, h), resample=Image.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2.0 * image - 1.0


def preprocess_mask(mask, smoothing=None):
    """Preprocess an inpainting mask"""
    mask = mask.convert("L")
    if smoothing is not None:
        smoothed = mask.filter(ImageFilter.GaussianBlur(smoothing))
        mask = Image.composite(mask, smoothed, mask)  # Original mask values kept as 1, out of mask get smoothed
        mask.save("outputs/smoothed_mask.png")  # FIXME
    w, h = mask.size
    w, h = map(lambda x: x - x % 32, (w, h))  # resize to integer multiple of 32
    mask = mask.resize((w // 8, h // 8), resample=Image.NEAREST)
    mask = np.array(mask).astype(np.float32) / 255.0
    mask = np.tile(mask, (4, 1, 1))
    mask = mask[None].transpose(0, 1, 2, 3)  # what does this step do?
    mask = 1 - mask  # repaint white, keep black
    mask = torch.from_numpy(mask)
    return mask
