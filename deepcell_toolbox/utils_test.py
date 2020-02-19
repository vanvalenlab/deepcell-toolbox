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
"""Tests for post-processing functions"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage.measure import label

import pytest

from deepcell_toolbox import utils


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


def _generate_test_masks():
    img_w = img_h = 30
    mask_images = []
    for _ in range(8):
        imarray = np.random.randint(2, size=(img_w, img_h, 1))
        mask_images.append(imarray)
    return mask_images


def test_erode_edges_2d():
    for img in _generate_test_masks():
        img = label(img)
        img = np.squeeze(img)

        erode_0 = utils.erode_edges(img, erosion_width=0)
        erode_1 = utils.erode_edges(img, erosion_width=1)
        erode_2 = utils.erode_edges(img, erosion_width=2)

        assert img.shape == erode_0.shape
        assert erode_0.shape == erode_1.shape
        assert erode_1.shape == erode_2.shape
        np.testing.assert_equal(erode_0, img)
        assert np.sum(erode_0) > np.sum(erode_1)
        assert np.sum(erode_1) > np.sum(erode_2)

        # test too few dims
        with pytest.raises(ValueError):
            erode_1 = utils.erode_edges(img[0], erosion_width=1)


def test_erode_edges_3d():
    mask_stack = np.array(_generate_test_masks())
    unique = np.zeros(mask_stack.shape)

    for i, mask in enumerate(_generate_test_masks()):
        unique[i] = label(mask)

    unique = np.squeeze(unique)

    erode_0 = utils.erode_edges(unique, erosion_width=0)
    erode_1 = utils.erode_edges(unique, erosion_width=1)
    erode_2 = utils.erode_edges(unique, erosion_width=2)

    assert unique.shape == erode_0.shape
    assert erode_0.shape == erode_1.shape
    assert erode_1.shape == erode_2.shape
    np.testing.assert_equal(erode_0, unique)
    assert np.sum(erode_0) > np.sum(erode_1)
    assert np.sum(erode_1) > np.sum(erode_2)

    # test too many dims
    with pytest.raises(ValueError):
        unique = np.expand_dims(unique, axis=-1)
        erode_1 = utils.erode_edges(unique, erosion_width=1)


def test_correct_drift():
    img2d = np.random.rand(30, 30)
    img3d = np.random.rand(10, 30, 30)
    img4d = np.random.rand(10, 30, 30, 1)

    # Wrong  input size
    with pytest.raises(ValueError):
        utils.correct_drift(img2d)

    # Mismatched inputs
    with pytest.raises(ValueError):
        utils.correct_drift(img3d, img4d)

    # 3d X alone
    res = utils.correct_drift(img3d)
    assert len(res.shape) == 3

    # 3d with y
    res = utils.correct_drift(img3d, img3d)
    assert len(res) == 2
    assert len(res[0].shape) == 3
    assert len(res[1].shape) == 3

    # 4d input
    res = utils.correct_drift(img4d)
    assert len(res.shape) == 4


def test_tile_image():
    shape = (4, 20, 20, 1)
    big_image = np.random.random(shape)
    model_input_shape = (5, 5)
    stride_ratio = 0.8
    tiles, tiles_info = utils.tile_image(big_image, model_input_shape,
                                         stride_ratio=stride_ratio)

    assert tiles_info['image_shape'] == shape

    expected_batches = shape[0]
    expected_batches *= (shape[1] // model_input_shape[0]) / stride_ratio
    expected_batches *= (shape[2] // model_input_shape[1]) / stride_ratio
    assert tiles.shape[0] == int(expected_batches)  # pylint: disable=E1136


def test_untile_image():
    shape = (4, 20, 20, 1)
    big_image = np.random.random(shape)
    model_input_shape = (5, 5)
    stride_ratio = 0.75
    tiles, tiles_info = utils.tile_image(big_image, model_input_shape,
                                         stride_ratio=stride_ratio)

    untiled_image = utils.untile_image(tiles, tiles_info, model_input_shape)
    np.testing.assert_equal(untiled_image, big_image)
