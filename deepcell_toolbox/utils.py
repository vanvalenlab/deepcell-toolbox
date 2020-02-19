# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-data-processing/LICENSE
#
# The Work provided may be used for non-commercial academic purposes only.
# For any other use of the Work, including commercial use, please contact:
# vanvalenlab@gmail.com
#
# Neither the name of Caltech nor the names of its contributors may be used
# to endorse or promote products derived from this software without specific
# prior written permission.
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
"""Utility functions that may be used in other transforms."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.ndimage import fourier_shift
from skimage.morphology import ball, disk
from skimage.morphology import binary_erosion
from skimage.feature import register_translation


def erode_edges(mask, erosion_width):
    """Erode edge of objects to prevent them from touching

    Args:
        mask (numpy.array): uniquely labeled instance mask
        erosion_width (int): integer value for pixel width to erode edges

    Returns:
        numpy.array: mask where each instance has had the edges eroded

    Raises:
        ValueError: mask.ndim is not 2 or 3
    """
    if erosion_width:
        new_mask = np.zeros(mask.shape)
        if mask.ndim == 2:
            strel = disk(erosion_width)
        elif mask.ndim == 3:
            strel = ball(erosion_width)
        else:
            raise ValueError('erode_edges expects arrays of ndim 2 or 3.'
                             'Got ndim: {}'.format(mask.ndim))
        for cell_label in np.unique(mask):
            if cell_label != 0:
                temp_img = mask == cell_label
                temp_img = binary_erosion(temp_img, strel)
                new_mask = np.where(mask == cell_label, temp_img, new_mask)
        return np.multiply(new_mask, mask).astype('int')
    return mask


def correct_drift(X, y=None):
    """Correct drift across frames of numpy arrays.

    Args:
        X (numpy.array): The raw data to correct.
        y (numpy.array): Optional, the labeled data to correct.

    Returns:
        numpy.array: The drift-corrected data.

    Raises:
        ValueError: If X.dim != 3.
    """
    if len(X.shape) < 3:
        raise ValueError('A minimum of 3 dimensons are required.'
                         'Found {} dimensions.'.format(len(X.shape)))

    if y is not None and len(X.shape) != len(y.shape):
        raise ValueError('y {} must have same shape as X {}'.format(y.shape, X.shape))

    def _shift_image(img, shift):
        # Shift frame
        img_corr = fourier_shift(np.fft.fftn(img), shift)
        img_corr = np.fft.ifftn(img_corr)

        # Set values offset by shift to zero
        if shift[0] < 0:
            img_corr[int(shift[0]):, :] = 0
        elif shift[0] > 0:
            img_corr[:int(shift[0]), :] = 0

        if shift[1] < 0:
            img_corr[:, int(shift[1]):] = 0
        elif shift[1] > 0:
            img_corr[:, :int(shift[1])] = 0

        return img_corr

    # Start with the first image since we compare to the previous
    for t in range(1, X.shape[0]):
        # Calculate shift
        shift, _, _ = register_translation(X[t - 1], X[t])

        # Correct X image
        X[t] = _shift_image(X[t], shift)

        # Correct y if available
        if y is not None:
            y[t] = _shift_image(y[t], shift)

    if y is not None:
        return X, y

    return X


def tile_image(image, model_input_shape=(512, 512), stride_ratio=0.75):
    """
    Tile large image into many overlapping tiles of size "model_input_shape".

    Args:
        image (numpy.array): The image to tile.
        model_input_shape (tuple): The input size of the model.
        stride_ratio (float): The ratio of overlap between stride
            and tile shape.

    Returns:
        tuple(numpy.array, dict): An tuple consisting of an array of tiled
            images and a dictionary of tiling details (for use in un-tiling).
    """
    image_size_x, image_size_y = image.shape[1:3]
    tile_size_x = model_input_shape[0]
    tile_size_y = model_input_shape[1]

    stride_x = np.int(stride_ratio * tile_size_x)
    stride_y = np.int(stride_ratio * tile_size_y)

    rep_number_x = np.int(np.ceil((image_size_x - tile_size_x) / stride_x + 1))
    rep_number_y = np.int(np.ceil((image_size_y - tile_size_y) / stride_y + 1))
    new_batch_size = image.shape[0] * rep_number_x * rep_number_y

    tiles_shape = (new_batch_size, tile_size_x, tile_size_y, image.shape[3])
    tiles = np.zeros(tiles_shape, dtype=image.dtype)

    counter = 0
    batches = []
    x_starts = []
    x_ends = []
    y_starts = []
    y_ends = []

    for b in range(image.shape[0]):
        for i in range(rep_number_x):
            for j in range(rep_number_y):
                x_axis = 1
                if i != rep_number_x - 1:  # not the last one
                    x_start, x_end = i * stride_x, i * stride_x + tile_size_x
                else:
                    x_start, x_end = -tile_size_x, image.shape[x_axis]

                if j != rep_number_y - 1:  # not the last one
                    y_start, y_end = j * stride_y, j * stride_y + tile_size_y
                else:
                    y_start, y_end = -tile_size_y, image.shape[x_axis + 1]

                tiles[counter] = image[b, x_start:x_end, y_start:y_end, :]
                batches.append(b)
                x_starts.append(x_start)
                x_ends.append(x_end)
                y_starts.append(y_start)
                y_ends.append(y_end)
                counter += 1

    tiles_info = {}
    tiles_info['batches'] = batches
    tiles_info['x_starts'] = x_starts
    tiles_info['x_ends'] = x_ends
    tiles_info['y_starts'] = y_starts
    tiles_info['y_ends'] = y_ends
    tiles_info['stride_x'] = stride_x
    tiles_info['stride_y'] = stride_y
    tiles_info['image_shape'] = image.shape
    tiles_info['dtype'] = image.dtype

    return tiles, tiles_info


def untile_image(tiles, tiles_info,
                 model_input_shape=(512, 512)):
    """Untile a set of tiled images back to the original model shape.

    Args:
        tiles (numpy.array): The tiled images image to untile.
        tiles_info (dict): Details of how the image was tiled (from tile_image).
        model_input_shape (tuple): The input size of the model.

    Returns:
        numpy.array: The untiled image.
    """
    _axis = 1
    image_shape = tiles_info['image_shape']
    batches = tiles_info['batches']
    x_starts = tiles_info['x_starts']
    x_ends = tiles_info['x_ends']
    y_starts = tiles_info['y_starts']
    y_ends = tiles_info['y_ends']
    stride_x = tiles_info['stride_x']
    stride_y = tiles_info['stride_y']
    dtype = tiles_info['dtype']

    tile_size_x = model_input_shape[0]
    tile_size_y = model_input_shape[1]

    image_shape = tuple(list(image_shape[0:3]) + [tiles.shape[-1]])
    image = np.zeros(image_shape, dtype=dtype)

    zipped = zip(tiles, batches, x_starts, x_ends, y_starts, y_ends)
    for tile, batch, x_start, x_end, y_start, y_end in zipped:
        tile_x_start = 0
        tile_x_end = tile_size_x
        tile_y_start = 0
        tile_y_end = tile_size_y

        if x_start != 0:
            x_start += (tile_size_x - stride_x) / 2
            tile_x_start += (tile_size_x - stride_x) / 2
        if x_end != image_shape[_axis]:
            x_end -= (tile_size_x - stride_x) / 2
            tile_x_end -= (tile_size_x - stride_x) / 2
        if y_start != 0:
            y_start += (tile_size_y - stride_y) / 2
            tile_y_start += (tile_size_y - stride_y) / 2
        if y_end != image_shape[_axis]:
            y_end -= (tile_size_y - stride_y) / 2
            tile_y_end -= (tile_size_y - stride_y) / 2

        x_start = np.int(x_start)
        x_end = np.int(x_end)
        y_start = np.int(y_start)
        y_end = np.int(y_end)

        tile_x_start = np.int(tile_x_start)
        tile_x_end = np.int(tile_x_end)
        tile_y_start = np.int(tile_y_start)
        tile_y_end = np.int(tile_y_end)

        image[batch, x_start:x_end, y_start:y_end, :] = \
            tile[tile_x_start:tile_x_end, tile_y_start:tile_y_end, :]

    return image
