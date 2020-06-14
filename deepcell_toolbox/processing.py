# Copyright 2016-2019 The Van Valen Lab at the California Institute of
# Technology (Caltech), with support from the Paul Allen Family Foundation,
# Google, & National Institutes of Health (NIH) under Grant U24CA224309-01.
# All rights reserved.
#
# Licensed under a modified Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.github.com/vanvalenlab/deepcell-toolbox/LICENSE
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
from skimage.exposure import equalize_adapthist
from skimage.exposure import rescale_intensity
from skimage.measure import label


def normalize(image):
    """Normalize image data by dividing by the maximum pixel value

    Args:
        image: numpy array of image data

    Returns:
        normal_image: normalized image data
    """
    normal_image = (image - image.mean()) / image.std()
    return normal_image


def histogram_norm_preprocess(image, kernel_size=64):
    """Pre-process images using Contrast Limited Adaptive
    Histogram Equalization (CLAHE).

    Args:
        image (numpy.array): numpy array of phase image data.
        kernel_size (integer): Size of kernel for CLAHE.

    Returns:
        numpy.array: Pre-processed image data with dtype float32.
    """
    if not np.issubdtype(image.dtype, np.floating):
        print('Converting image dtype to float')
    image = image.astype('float32')

    for batch in range(image.shape[0]):
        for channel in range(image.shape[-1]):
            X = image[batch, ..., channel]
            X = rescale_intensity(X, out_range='float')
            X = equalize_adapthist(X, kernel_size=(kernel_size, kernel_size))
            image[batch, ..., channel] = X
    return image


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
