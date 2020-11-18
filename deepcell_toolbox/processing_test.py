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
"""Tests for post-processing functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import pytest

from deepcell_toolbox import processing


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


def test_normalize():
    height, width = 300, 300
    img = _get_image(height, width)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    normalized_img = processing.normalize(img)
    np.testing.assert_almost_equal(normalized_img.mean(), 0)
    np.testing.assert_almost_equal(normalized_img.var(), 1)

    # test single-valued image is non NaN.
    img = np.ones((height, width))
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)
    normalized_img = processing.normalize(img)
    np.testing.assert_almost_equal(normalized_img.mean(), 0)


def test_histogram_normalization():
    height, width = 30, 30
    image = _get_image(height, width)

    # make rank 4 (batch, X, y, channel)
    image = np.expand_dims(image, axis=0)
    image = np.expand_dims(image, axis=-1)

    # randomly flip sign of image values
    negative_filter = (2 * np.random.randint(0, 2, size=image.shape) - 1)

    # create a few other test inputs
    test_images = [
        image,
        image.astype('uint16'),
        image.astype('int16'),
        image.astype('float16'),
        image * negative_filter,
        image.astype('int16') * negative_filter,
        image.astype('float16') * negative_filter
    ]

    for img in test_images:
        preprocessed = processing.histogram_normalization(img)

        # test min and max values of output
        assert preprocessed.min() == 0 and preprocessed.max() == 1

        # test minima and maxima are consistent
        # min_coords = (img == img.min()).nonzero()
        # max_coords = (img == img.max()).nonzero()
        # assert (preprocessed[min_coords] == 0).all()
        # assert (preprocessed[max_coords] == 1).all()

        # test negative coordinates don't get clipped
        negative_coords = (img < 0).nonzero()
        if len(preprocessed[negative_coords]):
            assert (preprocessed[negative_coords] >= 0).all()

        # test legacy version
        legacy_img = processing.phase_preprocess(img)
        np.testing.assert_equal(legacy_img, preprocessed)


def test_percentile_threshold():
    image_data = np.random.rand(5, 20, 20, 2)
    image_data[4, 19, 4, 0] = 100

    thresholded = processing.percentile_threshold(image=image_data)
    assert np.all(thresholded < 100)

    # setting percentile to 100 shouldn't change data
    no_threshold = processing.percentile_threshold(image=image_data, percentile=100)
    assert np.array_equal(image_data, no_threshold)

    # different channels have different distributions
    image_data[:, :, :, 0] *= 100
    thresholded = processing.percentile_threshold(image=image_data)

    assert np.mean(thresholded[..., 0]) > 10
    assert np.mean(thresholded[..., 1]) < 1

    # blank channels are returned as blank
    image_data[0, ..., 0] = 0
    thresholded_blank = processing.percentile_threshold(image=image_data)
    assert np.all(thresholded_blank[0, ..., 0] == 0)


def test_mibi():
    channels = 3
    img = np.random.rand(300, 300, channels)
    mibi_img = processing.mibi(img)
    np.testing.assert_equal(mibi_img.shape, (300, 300, 1))


def test_pixelwise():
    channels = 4
    img = np.random.rand(1, 300, 300, channels)
    pixelwise_img = processing.pixelwise(img)
    assert pixelwise_img.shape == img.shape[:-1] + (1,)


def test_watershed():
    channels = np.random.randint(4, 8)
    img = np.random.rand(1, 300, 300, channels)
    watershed_img = processing.watershed(img)
    assert watershed_img.shape == img.shape[:-1] + (1,)


def test_correct_drift():
    img2d = np.random.rand(30, 30)
    img3d = np.random.rand(10, 30, 30)
    img4d = np.random.rand(10, 30, 30, 1)

    # Wrong  input size
    with pytest.raises(ValueError):
        processing.correct_drift(img2d)

    # Mismatched inputs
    with pytest.raises(ValueError):
        processing.correct_drift(img3d, img4d)

    # 3d X alone
    res = processing.correct_drift(img3d)
    assert len(res.shape) == 3

    # 3d with y
    res = processing.correct_drift(img3d, img3d)
    assert len(res) == 2
    assert len(res[0].shape) == 3
    assert len(res[1].shape) == 3

    # 4d input
    res = processing.correct_drift(img4d)
    assert len(res.shape) == 4
