import logging
import os

import torch
from scipy import ndimage
import numpy as np


class RunningAverage:
    """Computes and stores the average
    """

    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0

    def update(self, value, n=1):
        self.count += n
        self.sum += value * n
        self.avg = self.sum / self.count


def expand_as_one_hot(input, C, ignore_index=None):
    """
    Converts NxDxHxW label image to NxCxDxHxW, where each label gets converted to its corresponding one-hot vector
    :param input: 4D input image (NxDxHxW)
    :param C: number of channels/labels
    :param ignore_index: ignore index to be kept during the expansion
    :return: 5D output image (NxCxDxHxW)
    """
    # Convert from Spatial to 1xSpatial
    if input.dim() == 3:
        input = input.unsqueeze(0)
    assert input.dim() == 4

    # expand the input tensor to Nx1xDxHxW before scattering
    input = input.unsqueeze(1)
    # create result tensor shape (NxCxDxHxW)
    shape = list(input.size())
    shape[1] = C

    if ignore_index is not None:
        # create ignore_index mask for the result
        mask = input.expand(shape) == ignore_index
        # clone the src tensor and zero out ignore_index in the input
        input = input.clone()
        input[input == ignore_index] = 0
        # scatter to get the one-hot tensor
        result = torch.zeros(shape).to(input.device).scatter_(1, input, 1)
        # bring back the ignore_index in the result
        result[mask] = ignore_index
        return result
    else:
        # scatter to get the one-hot tensor
        return torch.zeros(shape).to(input.device).scatter_(1, input, 1)


def binarize_probabilities(output, class_dim=1):
    _, max_index = torch.max(output, dim=class_dim, keepdim=True)
    return torch.zeros_like(output, dtype=torch.int8, device=output.device).scatter_(class_dim, max_index, 1)


def flatten(tensor):
    """Flattens a given tensor such that the channel axis is first.
    The shapes are transformed as follows:
       (N, C, D, H, W) -> (C, N * D * H * W)
    """
    # number of channels
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)


def extract_largest_component(image: torch.Tensor):
    """
    :param image: Tensor of C x Spatial
    :return: Tensor of C x Spatial
    """
    device = image.device
    dtype = image.dtype
    image = image.cpu().numpy()

    label, num_label = ndimage.label(image == 1)
    size = np.bincount(label.ravel())
    biggest_label = size[1:].argmax() + 1
    return torch.from_numpy(label == biggest_label).to(device, dtype=dtype)


def count_percentage(tensor, name):
    matrix = tensor.cpu().numpy()
    logging.info(f'{name}: {np.count_nonzero(matrix) / np.size(matrix)} Shape: {matrix.shape}')


def make_dir(path):
    try:
        os.mkdir(path)
        print(f'Created {path} directory')
    except OSError:
        print(f'Directory {path} already exist')
