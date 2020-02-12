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
from skimage.measure import regionprops
import pytest

from deepcell_data_processing import processing


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


def _sample1(w, h, imw, imh):
    """Basic single cell synthetic sample"""
    x = np.random.randint(0, imw - w * 2)
    y = np.random.randint(0, imh - h * 2)

    im = np.zeros((imw, imh))
    im[x:x + w, y:y + h] = 1

    # Randomly rotate to pick horizontal or vertical
    if np.random.random() > 0.5:
        im = np.rot90(im)

    return im


def _retinanet_data(im):
    n_batch = 1
    n_det = 1
    mask_size = 14  # Is this correct?
    n_labels = 1

    # boxes
    rp = regionprops(im.astype(int))[0].bbox
    boxes = np.zeros((n_batch, n_det, 4))
    boxes[0, 0, :] = rp

    # scores
    scores = np.zeros((n_batch, n_det, 1))
    scores[0, 0, 0] = np.random.rand()

    # labels
    labels = np.zeros((n_batch, n_det, n_labels))
    labels[0, 0, 0] = 1

    # masks
    masks = np.ones((n_batch, n_det, mask_size, mask_size))

    # semantic
    semantic = np.zeros((n_batch, im.shape[0], im.shape[1], 4))
    semantic[:, :, :] = processing.watershed(np.reshape(
        im, (1, im.shape[0], im.shape[1], 1)))

    return [boxes, scores, labels, masks, semantic]


def test_normalize():
    height, width = 300, 300
    img = _get_image(height, width)
    normalized_img = processing.normalize(img)
    np.testing.assert_almost_equal(normalized_img.mean(), 0)
    np.testing.assert_almost_equal(normalized_img.var(), 1)


def test_mibi():
    channels = 3
    img = np.random.rand(300, 300, channels)
    mibi_img = processing.mibi(img)
    np.testing.assert_equal(mibi_img.shape, (300, 300, 1))


def test_pixelwise():
    channels = 4
    img = np.random.rand(300, 300, channels)
    pixelwise_img = processing.pixelwise(img)
    np.testing.assert_equal(pixelwise_img.shape, (300, 300, 1))


def test_watershed():
    channels = np.random.randint(4, 8)
    img = np.random.rand(300, 300, channels)
    watershed_img = processing.watershed(img)
    np.testing.assert_equal(watershed_img.shape, (300, 300, 1))


def test_retinanet():
    im = _sample1(10, 10, 40, 40)
    out = _retinanet_data(im)[:-1]

    label = processing.retinanet_to_label_image(out, 40, 40)


def test_retinanet_semantic():
    im = _sample1(10, 10, 40, 40)
    out = _retinanet_data(im)

    label = processing.retinanet_semantic_to_label_image(out)


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
