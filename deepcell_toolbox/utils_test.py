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

from itertools import product

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
    shapes = [
        (4, 21, 21, 1),
        (4, 21, 31, 2),
        (4, 31, 21, 3),
    ]
    model_input_shapes = [(3, 3), (5, 5), (7, 7), (12, 12)]

    stride_ratios = [0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 1]

    dtypes = ['int32', 'float32', 'uint16', 'float16']

    prod = product(shapes, model_input_shapes, stride_ratios, dtypes)

    for shape, input_shape, stride_ratio, dtype in prod:
        big_image = (np.random.random(shape) * 100).astype(dtype)
        tiles, tiles_info = utils.tile_image(
            big_image, input_shape,
            stride_ratio=stride_ratio)

        assert tiles_info['image_shape'] == shape
        assert tiles.shape[1:] == input_shape + (shape[-1],)
        assert tiles.dtype == big_image.dtype

        x_diff = shape[1] - input_shape[0]
        y_diff = shape[2] - input_shape[1]
        x_ratio = np.ceil(stride_ratio * input_shape[0])
        y_ratio = np.ceil(stride_ratio * input_shape[1])

        expected_batches = shape[0]
        expected_batches *= np.ceil(x_diff / x_ratio + 1)
        expected_batches *= np.ceil(y_diff / y_ratio + 1)
        expected_batches = int(expected_batches)
        # pylint: disable=E1136
        assert tiles.shape[0] == expected_batches

    # test bad input shape
    bad_shape = (21, 21, 1)
    bad_image = (np.random.random(bad_shape) * 100)
    with pytest.raises(ValueError):
        utils.tile_image(bad_image, (5, 5), stride_ratio=0.75)


def test_untile_image():
    shapes = [
        (4, 21, 21, 1),
        (4, 21, 31, 2),
        (4, 31, 21, 3),
    ]
    model_input_shapes = [(3, 3), (5, 5), (7, 7), (12, 12)]

    stride_ratios = [0.25, 0.33, 0.5, 0.66, 0.75, 0.8, 1]

    dtypes = ['int32', 'float32', 'uint16', 'float16']

    prod = product(shapes, model_input_shapes, stride_ratios, dtypes)

    for shape, input_shape, stride_ratio, dtype in prod:
        big_image = (np.random.random(shape) * 100).astype(dtype)
        tiles, tiles_info = utils.tile_image(
            big_image, input_shape,
            stride_ratio=stride_ratio)

        untiled_image = utils.untile_image(
            tiles=tiles, tiles_info=tiles_info,
            model_input_shape=input_shape)

        assert untiled_image.dtype == dtype
        assert untiled_image.shape == shape
        np.testing.assert_equal(untiled_image, big_image)


def test_resize():

    base_shape = [30, 30]
    out_shapes = [[50, 50], [10, 10]]
    channel_sizes = (1, 3)

    for out in out_shapes:
        for c in channel_sizes:
            # batch, channel first
            in_shape = [c] + base_shape + [4]
            out_shape = tuple([c] + out + [4])
            rs = utils.resize(np.random.rand(*in_shape), out, data_format='channels_first')
            assert out_shape == rs.shape

            # batch, channel last
            in_shape = [4] + base_shape + [c]
            out_shape = tuple([4] + out + [c])
            rs = utils.resize(np.random.rand(*in_shape), out, data_format='channels_last')
            assert out_shape == rs.shape

            # no batch, channel first
            in_shape = [c] + base_shape
            out_shape = tuple([c] + out)
            rs = utils.resize(np.random.rand(*in_shape), out, data_format='channels_first')
            assert out_shape == rs.shape

            # no batch, channel last
            in_shape = base_shape + [c]
            out_shape = tuple(out + [c])
            rs = utils.resize(np.random.rand(*in_shape), out, data_format='channels_last')
            assert out_shape == rs.shape

            # make sure label data is not linearly interpolated and returns only the same ints

            # no batch, channel last
            in_shape = base_shape + [c]
            out_shape = tuple(out + [c])
            in_data = np.random.choice(a=[0, 1, 9, 20], size=in_shape, replace=True)
            rs = utils.resize(in_data, out, data_format='channels_last', labeled_image=True)
            assert out_shape == rs.shape
            assert np.all(rs == np.floor(rs))
            assert np.all(np.unique(rs) == [0, 1, 9, 20])

            # batch, channel first
            in_shape = [c] + base_shape + [4]
            out_shape = tuple([c] + out + [4])
            in_data = np.random.choice(a=[0, 1, 9, 20], size=in_shape, replace=True)
            rs = utils.resize(in_data, out, data_format='channels_first', labeled_image=True)
            assert out_shape == rs.shape
            assert np.all(rs == np.floor(rs))
            assert np.all(np.unique(rs) == [0, 1, 9, 20])

    # Wrong data size
    with pytest.raises(ValueError):
        im = np.random.rand(20, 20)
        out_shape = (10, 10)
        rs = utils.resize(im, out_shape)

    # Wrong shape
    with pytest.raises(ValueError):
        im = np.random.rand(20, 20, 1)
        out_shape = (10, 10, 1)
        rs = utils.resize(im, out_shape, data_format='channels_last')
