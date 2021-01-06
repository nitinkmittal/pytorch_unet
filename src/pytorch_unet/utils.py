from torch import Tensor
import numpy as np


def inverse_transform(img: Tensor) -> np.ndarray:
    """
    Return numpy image array.

    Parameters
    ----------
    img: torch.Tensor of shape (n_channels, height, width)
        img is assumed to contain values between 0 and 1.

    Returns
    -------
    img: np.ndarray of shape (height, width, n_channels)
    """
    img = img.data.numpy().transpose((1, 2, 0))
    img = np.clip(img, 0, 1)
    img = (img * 255).astype(np.uint8)
    return img