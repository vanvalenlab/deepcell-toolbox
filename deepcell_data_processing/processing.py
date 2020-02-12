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
"""Functions for pre- and post-processing image data"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy import ndimage
from scipy.ndimage import fourier_shift
from skimage import morphology
from skimage.feature import peak_local_max, register_translation
from skimage.measure import label

from keras_retinanet.utils.compute_overlap import compute_overlap


def normalize(image):
    """Normalize image data by dividing by the maximum pixel value

    Args:
        image: numpy array of image data

    Returns:
        normal_image: normalized image data
    """
    normal_image = (image - image.mean()) / image.std()
    return normal_image


def mibi(prediction, edge_threshold=.25, interior_threshold=.25):
    """Post-processing for MIBI data. Uniquely segments every cell by
    repeatedly eroding and dilating the cell interior prediction  until a
    boundary is reached.

    Args:
        prediction: output from a pixelwise transform (edge, interior, bg)
        edge_threshold: confidence threshold to determine edge pixels
        interior_threshold: confidence threshold to determine interior pixels

    Returns:
        transformed data where each cell is labeled uniquely
    """

    def dilate(array, mask, num_dilations):
        copy = np.copy(array)
        for _ in range(0, num_dilations):
            dilated = morphology.dilation(copy)
            # dilate if still in mask range not in another cell
            copy = np.where((mask != 0) & (dilated != copy) & (copy == 0),
                            dilated, copy)
        return copy

    def dilate_nomask(array, num_dilations):
        copy = np.copy(array)
        for _ in range(0, num_dilations):
            dilated = morphology.dilation(copy)
            # if one cell not eating another, dilate
            copy = np.where((dilated != copy) & (copy == 0), dilated, copy)
        return copy

    def erode(array, num_erosions):
        original = np.copy(array)
        for _ in range(0, num_erosions):
            eroded = morphology.erosion(np.copy(original))
            original[original != eroded] = 0
        return original

    # edge = (prediction[..., 0] >= edge_threshold).astype('int')
    edge = np.copy(prediction[..., 0])
    edge[edge < edge_threshold] = 0
    edge[edge >= edge_threshold] = 1

    # interior = (prediction[..., 1] >= interior_threshold).astype('int')
    interior = np.copy(prediction[..., 1])
    interior[interior >= interior_threshold] = 1
    interior[interior < interior_threshold] = 0

    # define foreground as the interior bounded by edge
    fg_thresh = np.logical_and(interior == 1, edge == 0)

    # remove small objects from the foreground segmentation
    fg_thresh = morphology.remove_small_objects(
        fg_thresh, min_size=50, connectivity=1)

    fg_thresh = np.expand_dims(fg_thresh, axis=-1)

    segments = label(np.squeeze(fg_thresh), connectivity=2)

    for _ in range(8):
        segments = dilate(segments, interior, 2)
        segments = erode(segments, 1)

    segments = dilate(segments, interior, 2)

    for _ in range(2):
        segments = dilate_nomask(segments, 1)
        segments = erode(segments, 2)

    segments = np.expand_dims(segments, axis=-1)
    return segments.astype('uint16')


def watershed(image, min_distance=10, threshold_abs=0.05):
    """Use the watershed method to identify unique cells based
    on their distance transform.

    # TODO: labels should be the fgbg output, NOT the union of distances

    Args:
        image: distance transform of image (model output)
        min_distance: minimum number of pixels separating peaks
        threshold_abs: minimum intensity of peaks

    Returns:
        image mask where each cell is annotated uniquely
    """
    distance = np.argmax(image, axis=-1)
    labels = (distance > 0).astype('int')

    local_maxi = peak_local_max(
        image[..., -1],
        min_distance=min_distance,
        threshold_abs=threshold_abs,
        indices=False,
        labels=labels,
        exclude_border=False)

    # markers = label(local_maxi)
    markers = ndimage.label(local_maxi)[0]
    segments = morphology.watershed(-distance, markers, mask=labels)
    results = np.expand_dims(segments, axis=-1)
    results = morphology.remove_small_objects(
        results, min_size=50, connectivity=1)
    return results


def pixelwise(prediction, threshold=.8):
    """Post-processing for pixelwise transform predictions.
    Uses the interior predictions to uniquely label every instance.

    Args:
        prediction: pixelwise transform prediction
        threshold: confidence threshold for interior predictions

    Returns:
        post-processed data with each cell uniquely annotated
    """
    if prediction.shape[0] == 1:
        prediction = np.squeeze(prediction, axis=0)
    interior = prediction[..., 2] > threshold
    data = np.expand_dims(interior, axis=-1)
    labeled = ndimage.label(data)[0]
    labeled = morphology.remove_small_objects(
        labeled, min_size=50, connectivity=1)
    return labeled


def correct_drift(X, y=None):

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
