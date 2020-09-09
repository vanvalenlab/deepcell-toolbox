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


def test_histogram_normalization():
    height, width = 300, 300
    img = _get_image(height, width)

    # make rank 4 (batch, X, y, channel)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    preprocessed_img = processing.histogram_normalization(img)
    assert (preprocessed_img <= 1).all() and (preprocessed_img >= -1).all()

    preprocessed_img = processing.histogram_normalization(img.astype('uint16'))
    assert (preprocessed_img <= 1).all() and (preprocessed_img >= -1).all()

    # test legacy version
    preprocessed_img = processing.phase_preprocess(img)
    assert (preprocessed_img <= 1).all() and (preprocessed_img >= -1).all()


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


def test_multiplex_preprocess():
    height, width = 300, 300
    img = _get_image(height, width)

    # make rank 4 (batch, X, y, channel)
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=-1)

    # single bright spot
    img[0, 200, 200, 0] = 5000

    # histogram normalized
    processed = processing.multiplex_preprocess(img)
    assert (processed <= 1).all() and (processed >= -1).all()

    # maxima is no longer significantly greater than rest of image
    new_spot_val = processed[0, 200, 200, 0]
    processed[0, 200, 200, 0] = 0.5
    next_max_val = np.max(processed)

    # difference between bright spot and next greatest value is essentially nothing
    assert np.round(new_spot_val / next_max_val, 1) == 1

    # histogram normalization without thresholding
    processed_hist = processing.multiplex_preprocess(img, threshold=False)
    assert (processed_hist <= 1).all() and (processed_hist >= -1).all()

    new_spot_val = processed_hist[0, 200, 200, 0]
    processed_hist[0, 200, 200, 0] = 0.5
    next_max_val = np.max(processed_hist)
    assert np.round(new_spot_val / next_max_val, 1) > 1

    # thresholding without histogram normalization
    processed_thresh = processing.multiplex_preprocess(img, normalize=False)
    assert not (processed_thresh <= 1).all()

    new_spot_val = processed_thresh[0, 200, 200, 0]
    processed_thresh[0, 200, 200, 0] = 0.5
    next_max_val = np.max(processed_thresh)
    assert np.round(new_spot_val / next_max_val, 1) == 1

    # no change to image
    not_processed = processing.multiplex_preprocess(img, normalize=False, threshold=False)
    assert np.all(not_processed == img)


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
