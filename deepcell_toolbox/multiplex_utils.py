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
"""Helper functions for Muliplex segmentation"""

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import numpy as np

from deepcell_toolbox.deep_watershed import deep_watershed_mibi
from deepcell_toolbox.processing import percentile_threshold
from deepcell_toolbox.processing import histogram_normalization


def multiplex_preprocess(image, **kwargs):
    """Preprocess input data for multiplex model.

    Args:
        image: array to be processed

    Returns:
        np.array: processed image array
    """
    output = np.copy(image)
    threshold = kwargs.get('threshold', True)
    if threshold:
        percentile = kwargs.get('percentile', 99.9)
        output = percentile_threshold(image=output, percentile=percentile)

    normalize = kwargs.get('normalize', True)
    if normalize:
        kernel_size = kwargs.get('kernel_size', 128)
        output = histogram_normalization(image=output, kernel_size=kernel_size)

    return output


def multiplex_postprocess(model_output, compartment='whole-cell', whole_cell_kwargs=None,
                          nuclear_kwargs=None):
    """Postprocess model output to generate predictions for distinct cellular compartments

    Args:
        model_output (dict): Output from deep watershed model. A dict with a key corresponding to
            each cellular compartment with a model prediction. Each key maps to a subsequent dict
            with the following keys entries
            - inner-distance: Prediction for the inner distance transform.
            - outer-distance: Prediction for the outer distance transform
            - fgbg-fg: prediction for the foreground/background transform
            - pixelwise-interior: Prediction for the interior/border/background transform.
        compartment: which cellular compartments to generate predictions for.
            must be one of 'whole_cell', 'nuclear', 'both'
        whole_cell_kwargs (dict): Optional list of post-processing kwargs for whole-cell prediction
        nuclear_kwargs (dict): Optional list of post-processing kwargs for nuclear prediction

    Returns:
        numpy.array: Uniquely labeled mask for each compartment

    Raises:
        ValueError: for invalid compartment flag
    """

    valid_compartments = ['whole-cell', 'nuclear', 'both']

    if whole_cell_kwargs is None:
        whole_cell_kwargs = {}

    if nuclear_kwargs is None:
        nuclear_kwargs = {}

    if compartment not in valid_compartments:
        raise ValueError('Invalid compartment supplied: {}. '
                         'Must be one of {}'.format(compartment, valid_compartments))

    if compartment == 'whole-cell':
        label_images = deep_watershed_mibi(model_output=model_output['whole-cell'],
                                           **whole_cell_kwargs)
    elif compartment == 'nuclear':
        label_images = deep_watershed_mibi(model_output=model_output['nuclear'],
                                           **nuclear_kwargs)
    elif compartment == 'both':
        label_images_cell = deep_watershed_mibi(model_output=model_output['whole-cell'],
                                                **whole_cell_kwargs)

        label_images_nucleus = deep_watershed_mibi(model_output=model_output['nuclear'],
                                                   **nuclear_kwargs)

        label_images = np.concatenate((label_images_cell, label_images_nucleus), axis=-1)

    else:
        raise ValueError('Invalid compartment supplied: {}. '
                         'Must be one of {}'.format(compartment, valid_compartments))

    return label_images


def format_output_multiplex(output_list):
    """Takes list of model outputs and formats into a dictionary for better readability

    Args:
        output_list (list): predictions from semantic heads

    Returns:
        dict: dictionary with predictions

    Raises:
        ValueError: if model output list is not len(8)
    """
    expected_length = 4
    if len(output_list) != expected_length:
        raise ValueError('output_list was length {}, expecting length {}'.format(
            len(output_list), expected_length))

    formatted_dict = {
        'whole-cell': {
            'inner-distance': output_list[0],
            'pixelwise-interior': output_list[1][..., 1:2]
        },
        'nuclear': {
            'inner-distance': output_list[2],
            'pixelwise-interior': output_list[3][..., 1:2]
        }
    }

    return formatted_dict
