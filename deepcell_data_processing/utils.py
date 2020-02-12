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

