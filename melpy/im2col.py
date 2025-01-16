import numpy as np
from melpy.tensor import *

def get_indices(image_shape, window_shape, stride):
    """
    Computes the indices for converting an image to a column matrix.

    Parameters
    ----------
    image_shape : tuple
        The shape of the input image (batch_size, channels, height, width).
    window_shape : int
        The size of the sliding window (kernel size).
    stride : int
        The stride of the sliding window.

    Returns
    -------
    k : ndarray
        Indices for the channel dimension.
    i : ndarray
        Indices for the height dimension.
    j : ndarray
        Indices for the width dimension.
    """
    output_height = int((image_shape[2] - window_shape + stride) // stride)
    output_width = int((image_shape[3] - window_shape + stride) // stride)

    level1 = np.repeat(np.arange(window_shape), window_shape)
    level1 = np.tile(level1, image_shape[1])

    increment = stride * np.repeat(np.arange(output_height), output_width)

    i = level1.reshape(-1, 1) + increment.reshape(1, -1)

    slide1 = np.tile(np.arange(window_shape), window_shape * image_shape[1])
    increment = stride * np.tile(np.arange(output_width), output_height)

    j = slide1.reshape(-1, 1) + increment.reshape(1, -1)

    k = np.repeat(np.arange(image_shape[1]), window_shape * window_shape).reshape(-1, 1)
    return k, i, j

def im2col(images, window_shape, stride):
    """
    Converts an image to a column matrix.

    Parameters
    ----------
    images : Tensor
        The input images with shape (batch_size, channels, height, width).
    window_shape : int
        The size of the sliding window (kernel size).
    stride : int
        The stride of the sliding window.

    Returns
    -------
    columns : Tensor
        The column matrix representation of the input images.
    """
    k, i, j = get_indices(images.shape, window_shape, stride)
    columns = Tensor(np.concatenate(images.array[:, k, i, j], axis=-1), requires_grad=True)
    return columns

def col2im(columns, image_shape, window_shape, stride):
    """
    Converts a column matrix back to an image.

    Parameters
    ----------
    columns : Tensor
        The column matrix representation of the images.
    image_shape : tuple
        The shape of the output image (batch_size, channels, height, width).
    window_shape : int
        The size of the sliding window (kernel size).
    stride : int
        The stride of the sliding window.

    Returns
    -------
    images : Tensor
        The reconstructed images with shape (batch_size, channels, height, width).
    """
    images = np.zeros(image_shape)
    k, i, j = get_indices(image_shape, window_shape, stride)
    cols_reshaped = np.array(np.hsplit(columns.array, image_shape[0]))
    np.add.at(images, (slice(None), k, i, j), cols_reshaped)
    return Tensor(images, requires_grad=True)
