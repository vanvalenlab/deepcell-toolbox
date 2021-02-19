# Copyright 2016-2021 The Van Valen Lab at the California Institute of
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

from deepcell_toolbox import deep_watershed


def _get_image(img_h=300, img_w=300):
    bias = np.random.rand(img_w, img_h) * 64
    variance = np.random.rand(img_w, img_h) * (255 - 64)
    img = np.random.rand(img_w, img_h) * variance + bias
    return img


def test_deep_watershed():
    shape = (5, 21, 21, 1)
    inner_distance = np.random.random(shape) * 100
    outer_distance = np.random.random(shape) * 100
    fgbg = np.random.randint(0, 1, size=shape)
    inputs = [inner_distance, outer_distance, fgbg]

    # basic tests
    watershed_img = deep_watershed.deep_watershed(inputs)
    np.testing.assert_equal(watershed_img.shape, shape[:-1])

    # turn some knobs
    watershed_img = deep_watershed.deep_watershed(inputs,
                                                  small_objects_threshold=1,
                                                  exclude_border=True)
    np.testing.assert_equal(watershed_img.shape, shape[:-1])


def test_deep_watershed_mibi():
    shape = (5, 21, 21, 1)
    inner_distance = np.random.random(shape) * 100
    outer_distance = np.random.random(shape) * 100
    fgbg = np.random.randint(0, 1, size=shape)
    pixelwise = np.random.random(shape[:-1] + (3, ))
    model_output = {'inner-distance': inner_distance,
                    'outer-distance': outer_distance,
                    'fgbg-fg': fgbg,
                    'pixelwise-interior': pixelwise}

    # basic tests
    watershed_img = deep_watershed.deep_watershed_mibi(model_output=model_output)
    np.testing.assert_equal(watershed_img.shape, shape)

    # turn some knobs
    watershed_img = deep_watershed.deep_watershed_mibi(model_output=model_output,
                                                       small_objects_threshold=1,
                                                       pixel_expansion=5)

    # turn turn
    watershed_img = deep_watershed.deep_watershed_mibi(model_output=model_output,
                                                       small_objects_threshold=1,
                                                       maxima_model='fgbg-fg',
                                                       interior_model='outer-distance',
                                                       maxima_model_smooth=0,
                                                       fill_holes_threshold=4)

    np.testing.assert_equal(watershed_img.shape, shape)

    for model_under_test in ['interior_model', 'maxima_model']:
        with pytest.raises(ValueError):
            bad_model = {model_under_test: 'bad_model_name'}
            watershed_img = deep_watershed.deep_watershed_mibi(model_output=model_output,
                                                               **bad_model)

    bad_array = pixelwise[..., 0]
    for bad_transform in ['inner-distance', 'pixelwise-interior']:
        bad_model_output = model_output.copy()
        bad_model_output[bad_transform] = bad_array
        with pytest.raises(ValueError):
            watershed_img = deep_watershed.deep_watershed_mibi(model_output=bad_model_output)


def test_deep_watershed_3D():
    shape = (5, 10, 21, 21, 1)
    inner_distance = np.random.random(shape) * 100
    outer_distance = np.random.random(shape) * 100
    fgbg = np.random.randint(0, 1, size=shape)
    inputs = [inner_distance, outer_distance, fgbg]

    # basic tests
    watershed_img = deep_watershed.deep_watershed_3D(inputs)
    np.testing.assert_equal(watershed_img.shape, shape[:-1])

    # turn some knobs
    watershed_img = deep_watershed.deep_watershed_3D(inputs,
                                                     small_objects_threshold=1,
                                                     exclude_border=True)
    np.testing.assert_equal(watershed_img.shape, shape[:-1])
