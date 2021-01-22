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
"""Functions for and post-processing RetinaNet outputs"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from skimage import morphology
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects
from skimage.segmentation import random_walker, relabel_sequential
from skimage.transform import resize

from deepcell_toolbox.compute_overlap import compute_overlap  # pylint: disable=E0401


def compute_iou(boxes, mask_image):
    overlaps = compute_overlap(boxes.astype('float64'), boxes.astype('float64'))
    ind_x, ind_y = np.nonzero(overlaps)
    ious = np.zeros(overlaps.shape)
    for index in range(ind_x.shape[0]):
        mask_a = mask_image[ind_x[index]]
        mask_b = mask_image[ind_y[index]]
        intersection = np.count_nonzero(np.logical_and(mask_a, mask_b))
        union = np.count_nonzero(mask_a + mask_b)
        if intersection > 0:
            ious[ind_x[index], ind_y[index]] = intersection / union

    return ious


def retinamask_semantic_postprocess(retinanet_outputs,
                                    score_threshold=0.5,
                                    multi_iou_threshold=0.25,
                                    binarize_threshold=0.5,
                                    watershed_threshold=0.5,
                                    perimeter_area_threshold=2,
                                    small_objects_threshold=100):

    boxes_batch = retinanet_outputs[-5]
    scores_batch = retinanet_outputs[-4]
    labels_batch = retinanet_outputs[-3]
    masks_batch = retinanet_outputs[-2]
    semantic_batch = retinanet_outputs[-1]

    # Create empty label matrix
    label_images = np.zeros(
        (masks_batch.shape[0], semantic_batch.shape[1], semantic_batch.shape[2]))

    # Iterate over batches
    for i in range(boxes_batch.shape[0]):
        boxes = boxes_batch[i]
        scores = scores_batch[i]
        labels = labels_batch[i]
        masks = masks_batch[i]
        semantic = semantic_batch[i]

        # Get good detections
        selection = np.nonzero(scores > score_threshold)[0]
        boxes = boxes[selection]
        scores = scores[selection]
        labels = labels[selection]
        masks = masks[selection, ..., -1]

        # Compute overlap of masks with each other
        mask_image = np.zeros((masks.shape[0], semantic.shape[0],
                               semantic.shape[1]), dtype='float32')

        for j in range(masks.shape[0]):
            mask = masks[j]
            box = boxes[j].astype(int)
            mask = resize(mask, (box[3] - box[1], box[2] - box[0]))
            mask = (mask > binarize_threshold).astype('float32')
            mask_image[j, box[1]:box[3], box[0]:box[2]] = mask

        ious = compute_iou(boxes, mask_image)

        # Identify all the masks with no overlaps and
        # add to the label matrix
        summed_ious = np.sum(ious, axis=-1)
        no_overlaps = np.where(summed_ious == 1)

        masks_no_overlaps = mask_image[no_overlaps]
        range_no_overlaps = np.arange(1, masks_no_overlaps.shape[0] + 1)
        masks_no_overlaps *= np.expand_dims(np.expand_dims(range_no_overlaps, axis=-1), axis=-1)

        masks_concat = masks_no_overlaps

        # If a mask has a big iou with two other masks, remove it
        overlaps = np.where(summed_ious > 1)
        bad_mask = np.sum(ious > multi_iou_threshold, axis=0)
        good_overlaps = np.logical_and(summed_ious > 1, bad_mask < 3)
        good_overlaps = np.where(good_overlaps == 1)

        # Identify all the ambiguous pixels and resolve
        # by performing marker based watershed using unambiguous
        # pixels as the markers
        masks_overlaps = mask_image[good_overlaps]
        range_overlaps = np.arange(1, masks_overlaps.shape[0] + 1)
        masks_overlaps_label = masks_overlaps * np.expand_dims(
            np.expand_dims(range_overlaps, axis=-1), axis=-1)

        masks_overlaps_sum = np.sum(masks_overlaps, axis=0)
        ambiguous_pixels = np.where(masks_overlaps_sum > 1)
        markers = np.sum(masks_overlaps_label, axis=0)

        if np.sum(markers.flatten()) > 0:
            markers[markers == 0] = -1
            markers[ambiguous_pixels] = 0

            foreground = masks_overlaps_sum > 0
            segments = random_walker(foreground, markers)

            masks_overlaps = np.zeros((np.amax(segments).astype(int),
                                       masks_overlaps.shape[1], masks_overlaps.shape[2]))

            for j in range(1, masks_overlaps.shape[0] + 1):
                masks_overlaps[j - 1] = segments == j

            range_overlaps = np.arange(masks_no_overlaps.shape[0] + 1,
                                       masks_no_overlaps.shape[0] + masks_overlaps.shape[0] + 1)
            masks_overlaps *= np.expand_dims(np.expand_dims(range_overlaps, axis=-1), axis=-1)
            masks_concat = np.concatenate([masks_concat, masks_overlaps], axis=0)

        # Find peaks in watershed that are not within any
        # box and perform watershed
        semantic_argmax = np.argmax(semantic, axis=-1)
        semantic_argmax *= np.sum(masks_concat, axis=0) < 1
        foreground = semantic_argmax > 0

        inner_most = semantic[..., -1] * (np.sum(masks_concat, axis=0) < 1)
        local_maxi = inner_most > watershed_threshold

        if np.sum(local_maxi.flatten()) > 0:
            markers_semantic = label(local_maxi)
            distance = semantic_argmax
            segments_semantic = morphology.watershed(
                -distance, markers_semantic, mask=foreground)

            # Remove misshapen watershed cells
            props = regionprops(segments_semantic)
            for prop in props:
                if prop.perimeter ** 2 / prop.area > perimeter_area_threshold * 4 * np.pi:
                    segments_semantic[segments_semantic == prop.label] = 0

            masks_semantic = np.zeros((np.amax(segments_semantic).astype(int),
                                       semantic.shape[0], semantic.shape[1]))

            for j in range(1, masks_semantic.shape[0] + 1):
                masks_semantic[j - 1] = segments_semantic == j

            range_semantic = np.arange(masks_no_overlaps.shape[0] + masks_overlaps.shape[0] + 1,
                                       masks_no_overlaps.shape[0] + masks_overlaps.shape[0] +
                                       masks_semantic.shape[0] + 1)
            masks_semantic *= np.expand_dims(np.expand_dims(range_semantic, axis=-1), axis=-1)

            masks_concat = np.concatenate([masks_concat, masks_semantic], axis=0)

        label_image = np.sum(masks_concat, axis=0).astype(int)

        # Remove small objects
        label_image = morphology.remove_small_objects(
            label_image, min_size=small_objects_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        # Store in batched array
        label_images[i] = label_image

    return label_images


def retinamask_postprocess(outputs,
                           image_shape=(256, 256),
                           score_threshold=0.5,
                           multi_iou_threshold=0.25,
                           binarize_threshold=0.5,
                           small_objects_threshold=0,
                           dtype='float32'):
    """Post processing function for RetinaMask models.
    Expects model that produces an output list [boxes, scores, labels, masks]

    Args:
        boxes_batch: Bounding box predictions.
        scores_batch: Scores for each detection.
        labels_batch: Label for each detection.
        masks_batch: Masks for each detection.
        image_shape: Shape of the image.
        score_threshold: Score threshold for detections.
        multi_iou_threshold: Threshold to suppress detections that have multiple
            overlaps with other detections.
        binarize_threshold: Threshold to binarize masks.
        small_objects_threshold: Area threshold to remove small objects.
        dtype: data type of label mask.

    Returns:
        numpy.array: label mask for the image.
    """
    boxes_batch = outputs[-4]
    scores_batch = outputs[-3]
    labels_batch = outputs[-2]
    masks_batch = outputs[-1]

    # Create empty label matrix
    label_images = np.zeros((masks_batch.shape[0], image_shape[0], image_shape[1]))

    # Iterate over batches
    for i in range(boxes_batch.shape[0]):
        boxes = boxes_batch[i]
        scores = scores_batch[i]
        labels = labels_batch[i]
        masks = masks_batch[i]

        # Get good detections
        selection = np.nonzero(scores > score_threshold)[0]
        boxes = boxes[selection]
        scores = scores[selection]
        labels = labels[selection]
        masks = masks[selection, ..., -1]

        # Compute overlap of masks with each other
        mask_shape = (masks.shape[0], image_shape[0], image_shape[1])
        mask_image = np.zeros(mask_shape, dtype=dtype)

        for j in range(masks.shape[0]):
            mask = masks[j]
            box = boxes[j].astype(int)
            if box[3] > box[1] and box[2] > box[0]:
                mask = resize(mask, (box[3] - box[1], box[2] - box[0]))
                mask = (mask > binarize_threshold).astype(dtype)
                mask_image[j, box[1]:box[3], box[0]:box[2]] = mask

        ious = compute_iou(boxes, mask_image)

        # Identify all the masks with no overlaps and
        # add to the label matrix
        summed_ious = np.sum(ious, axis=-1)
        no_overlaps = np.where(summed_ious == 1)

        masks_no_overlaps = mask_image[no_overlaps]
        range_no_overlaps = np.arange(1, masks_no_overlaps.shape[0] + 1)
        range_no_overlaps = np.expand_dims(range_no_overlaps, axis=-1)
        masks_no_overlaps *= np.expand_dims(range_no_overlaps, axis=-1)

        masks_concat = masks_no_overlaps

        # If a mask has a big iou with two other masks, remove it
        overlaps = np.where(summed_ious > 1)
        bad_mask = np.sum(ious > multi_iou_threshold, axis=0)
        good_overlaps = np.logical_and(summed_ious > 1, bad_mask < 3)
        good_overlaps = np.where(good_overlaps == 1)

        # Identify all the ambiguous pixels and resolve
        # by performing marker based watershed using unambiguous
        # pixels as the markers
        masks_overlaps = mask_image[good_overlaps]
        range_overlaps = np.arange(1, masks_overlaps.shape[0] + 1)
        masks_overlaps_label = masks_overlaps * np.expand_dims(
            np.expand_dims(range_overlaps, axis=-1), axis=-1)

        masks_overlaps_sum = np.sum(masks_overlaps, axis=0)
        ambiguous_pixels = np.where(masks_overlaps_sum > 1)
        markers = np.sum(masks_overlaps_label, axis=0)

        if np.sum(markers.flatten()) > 0:
            markers[markers == 0] = -1
            markers[ambiguous_pixels] = 0

            foreground = masks_overlaps_sum > 0
            segments = random_walker(foreground, markers)

            masks_overlaps = np.zeros((np.amax(segments).astype(int),
                                       masks_overlaps.shape[1], masks_overlaps.shape[2]))

            for j in range(1, masks_overlaps.shape[0] + 1):
                masks_overlaps[j - 1] = segments == j
            range_overlaps = np.arange(masks_no_overlaps.shape[0] + 1,
                                       masks_no_overlaps.shape[0] + masks_overlaps.shape[0] + 1)
            masks_overlaps *= np.expand_dims(np.expand_dims(range_overlaps, axis=-1), axis=-1)
            masks_concat = np.concatenate([masks_concat, masks_overlaps], axis=0)

        label_image = np.sum(masks_concat, axis=0).astype(int)

        # Remove small objects
        label_image = remove_small_objects(label_image, min_size=small_objects_threshold)

        # Relabel the label image
        label_image, _, _ = relabel_sequential(label_image)

        # Store in batched array
        label_images[i] = label_image

    return label_images
