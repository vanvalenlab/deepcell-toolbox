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


def deep_watershed_mibi(outputs,
                        min_distance=10,
                        detection_threshold=0.1,
                        distance_threshold=0.25,
                        exclude_border=False,
                        small_objects_threshold=0):
    """Postprocessing function for multiplexed deep watershed models. Thresholds the inner
    distance prediction to find cell centroids, which are used to seed a marker
    based watershed of the pixelwise interior prediction.

    Args:
        outputs (list): DeepWatershed model output. A list of
            [inner_distance, outer_distance, fgbg, pixelwise].

            - inner_distance: Prediction for the inner distance transform.
            - outer_distance: Prediction for the outer distance transform.
            - fgbg: Prediction for the foregound/background transform.
            - pixelwise: Prediction for the interior/border/background transform.

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
    pixel_interior_batch = outputs[3][:, ..., 1]

    label_images = []
    for batch in range(inner_distance_batch.shape[0]):
        inner_distance = inner_distance_batch[batch]
        inner_distance = nd.gaussian_filter(inner_distance, 3)
        pixel_interior = pixel_interior_batch[batch]
        pixel_interior = nd.gaussian_filter(pixel_interior, 2)

        markers = peak_local_max(inner_distance,
                                 min_distance=min_distance,
                                 threshold_abs=detection_threshold,
                                 exclude_border=exclude_border,
                                 indices=False)
        markers = label(markers)

        label_image = watershed(-pixel_interior,
                                markers,
                                mask=pixel_interior > distance_threshold,
                                watershed_line=0)

        # Remove small objects
        label_image = remove_small_objects(label_image, min_size=small_objects_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        label_images.append(label_image)

    label_images = np.stack(label_images, axis=0)

    return label_images
