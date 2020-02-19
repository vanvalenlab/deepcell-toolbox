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
