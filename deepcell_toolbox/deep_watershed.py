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
import scipy.ndimage as nd

from skimage.feature import peak_local_max
from skimage.measure import label
from skimage.morphology import watershed, remove_small_objects
from skimage.segmentation import relabel_sequential

from deepcell_toolbox.utils import erode_edges


def deep_watershed(outputs,
                   min_distance=10,
                   detection_threshold=0.1,
                   distance_threshold=0.01,
                   exclude_border=False,
                   small_objects_threshold=0):
    """Postprocessing function for deep watershed models. Thresholds the inner
    distance prediction to find cell centroids, which are used to seed a marker
    based watershed of the outer distance prediction.

    Args:
        outputs (list): DeepWatershed model output. A list of
            [inner_distance, outer_distance, fgbg].

            - inner_distance: Prediction for the inner distance transform.
            - outer_distance: Prediction for the outer distance transform.
            - fgbg: Prediction for the foregound/background transform.

        min_distance (int): Minimum allowable distance between two cells.
        detection_threshold (float): Threshold for the inner distance.
        distance_threshold (float): Threshold for the outer distance.
        exclude_border (bool): Whether to include centroid detections
            at the border.
        small_objects_threshold (int): Removes objects smaller than this size.

    Returns:
        numpy.array: Uniquely labeled mask.
    """
    inner_distance_batch = outputs[0][:, ..., 0]
    outer_distance_batch = outputs[1][:, ..., 0]

    label_images = []
    for batch in range(inner_distance_batch.shape[0]):
        inner_distance = inner_distance_batch[batch]
        outer_distance = outer_distance_batch[batch]

        coords = peak_local_max(inner_distance,
                                min_distance=min_distance,
                                threshold_abs=detection_threshold,
                                exclude_border=exclude_border)

        markers = np.zeros(inner_distance.shape)
        markers[coords[:, 0], coords[:, 1]] = 1
        markers = label(markers)
        label_image = watershed(-outer_distance,
                                markers,
                                mask=outer_distance > distance_threshold)
        label_image = erode_edges(label_image, 1)

        # Remove small objects
        label_image = remove_small_objects(label_image, min_size=small_objects_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)

    return label_images


def deep_watershed_mibi(model_output,
                        min_distance=10,
                        maxima_threshold=0.1,
                        cell_threshold=0.2,
                        exclude_border=False,
                        small_objects_threshold=0,
                        cell_model='pixelwise-interior',
                        maxima_model='inner-distance',
                        cell_model_smooth=1,
                        maxima_model_smooth=1):
    """Postprocessing function for multiplexed deep watershed models. Thresholds the inner
    distance prediction to find cell centroids, which are used to seed a marker
    based watershed of the pixelwise interior prediction.

    Args:
        model_output (dict): DeepWatershed model output. A dictionary containing key: value pairs
            with the transform name and the corresponding output. Currently supported keys:

            - inner_distance: Prediction for the inner distance transform.
            - outer_distance: Prediction for the outer distance transform.
            - fgbg: Foreground prediction for the foregound/background transform.
            - pixelwise_interior: Interior prediction for the interior/border/background transform.

        min_distance (int): Minimum allowable distance between two cell maxima.
        maxima_threshold (float): Threshold for the maxima prediction.
        cell_threshold (float): Threshold for the cell prediction.
        exclude_border (bool): Whether to include centroid detections at the border.
        small_objects_threshold (int): Removes objects smaller than this size.
        cell_model: semantic head to use to predict location of cells
        maxima_model: semantic head to use to predict maxima of cells
        cell_model_smooth: smoothing factor to apply to interior model predictions
        maxima_model_smooth: smoothing factor to apply to interior model predictions

    Returns:
        numpy.array: Uniquely labeled mask.
    """

    cell_model, maxima_model = cell_model.lower(), maxima_model.lower()

    valid_model_names = {'inner-distance', 'outer-distance', 'fgbg-fg', 'pixelwise-interior'}

    for name, model in zip(['cell_model', 'maxima_model'], [cell_model, maxima_model]):
        if model not in valid_model_names:
            raise ValueError('{} must be one of {}, got {}'.format(
                name, valid_model_names, model))

    cell_prediction_batch = model_output[cell_model]
    maxima_prediction_batch = model_output[maxima_model]

    zipped = zip(['cell_prediction', 'maxima_prediction'],
                 (cell_prediction_batch, maxima_prediction_batch))
    for name, arr in zipped:
        if len(arr.shape) != 4:
            raise ValueError('Model output must be of length 4. The {} model '
                             'provided was of shape {}'.format(name, arr.shape))

    label_images = []
    for batch in range(cell_prediction_batch.shape[0]):
        cell_prediction = cell_prediction_batch[batch, ..., 0]
        cell_prediction = nd.gaussian_filter(cell_prediction, cell_model_smooth)
        maxima_prediction = maxima_prediction_batch[batch, ..., 0]
        maxima_prediction = nd.gaussian_filter(maxima_prediction, maxima_model_smooth)

        markers = peak_local_max(maxima_prediction,
                                 min_distance=min_distance,
                                 threshold_abs=maxima_threshold,
                                 exclude_border=exclude_border,
                                 indices=False)
        markers = label(markers)

        label_image = watershed(-cell_prediction,
                                markers,
                                mask=cell_prediction > cell_threshold,
                                watershed_line=0)

        # Remove small objects
        label_image = remove_small_objects(label_image, min_size=small_objects_threshold)
        print(label_image.shape)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)
    label_images = np.expand_dims(label_images, axis=-1)

    return label_images


def deep_watershed_subcellular(model_output, compartment='whole-cell', whole_cell_kwargs=None,
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
        output_list: list of predictions from semantic heads

    Returns:
        formatted_dict: dictionary with predictions

    Raises: ValueError if model output list is not len(8)
    """

    if len(output_list) != 8:
        raise ValueError('output_list was length {}, expecting length 8'.format(len(output_list)))

    formatted_dict = {
        'whole-cell': {
            'inner-distance': output_list[0],
            'outer-distance': output_list[1],
            'fgbg-fg': output_list[2][..., :1],
            'pixelwise-interior': output_list[3][..., 1:2]
        },
        'nuclear': {
            'inner-distance': output_list[4],
            'outer-distance': output_list[5],
            'fgbg-fg': output_list[6][..., :1],
            'pixelwise-interior': output_list[7][..., 1:2]
        }
    }

    return formatted_dict
