from jupyter_bbox_widget import bbox
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import ot
import numpy as np
import pdb


def get_bbox_overlaps(bboxes1, bboxes2, mode="iou", eps=1e-6):
    if mode == "iou":
        return bbox_overlaps(bboxes1, bboxes2, eps=eps)
    elif mode == "giou":
        return bbox_gious(bboxes1, bboxes2, eps=eps)


def bbox_gious(bboxes1, bboxes2, eps=1e-6, use_legacy_coordinate=False):
    """Calculate the ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1 (ndarray): Shape (n, 4)
        bboxes2 (ndarray): Shape (k, 4)
        use_legacy_coordinate (bool): Whether to use coordinate system in
            mmdet v1.x. which means width, height should be
            calculated as 'x2 - x1 + 1` and 'y2 - y1 + 1' respectively.
            Note when function is used in `VOCDataset`, it should be
            True to align with the official implementation
            `http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCdevkit_18-May-2011.tar`
            Default: False.
    Returns:
        gious (ndarray): Shape (n, k)
    """

    if not use_legacy_coordinate:
        extra_length = 0.0
    else:
        extra_length = 1.0
    bboxes1 = bboxes1.astype(np.float32)
    bboxes2 = bboxes2.astype(np.float32)
    rows = bboxes1.shape[0]
    cols = bboxes2.shape[0]
    gious = np.zeros((rows, cols), dtype=np.float32)
    if rows * cols == 0:
        return gious
    exchange = False
    if bboxes1.shape[0] > bboxes2.shape[0]:
        bboxes1, bboxes2 = bboxes2, bboxes1
        gious = np.zeros((cols, rows), dtype=np.float32)
        exchange = True
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + extra_length) * (
        bboxes1[:, 3] - bboxes1[:, 1] + extra_length
    )
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + extra_length) * (
        bboxes2[:, 3] - bboxes2[:, 1] + extra_length
    )
    for i in range(bboxes1.shape[0]):
        x_start = np.maximum(bboxes1[i, 0], bboxes2[:, 0])
        y_start = np.maximum(bboxes1[i, 1], bboxes2[:, 1])
        x_end = np.minimum(bboxes1[i, 2], bboxes2[:, 2])
        y_end = np.minimum(bboxes1[i, 3], bboxes2[:, 3])
        overlap = np.maximum(x_end - x_start + extra_length, 0) * np.maximum(
            y_end - y_start + extra_length, 0
        )

        union = area1[i] + area2 - overlap
        union = np.maximum(union, eps)
        ious = overlap / union

        # Finding the coordinate of smallest enclosing box
        x_min = np.minimum(bboxes1[i, 0], bboxes2[:, 0])
        y_min = np.minimum(bboxes1[i, 1], bboxes2[:, 1])
        x_max = np.maximum(bboxes1[i, 2], bboxes2[:, 2])
        y_max = np.maximum(bboxes1[i, 3], bboxes2[:, 3])
        hull = (x_max - x_min + extra_length) * (y_max - y_min + extra_length)

        gious[i, :] = ious - (hull - union) / hull

    if exchange:
        gious = gious.T

    return gious


def add_label(result):
    labels = [[i] * len(r) for i, r in enumerate(result)]
    labels = np.hstack(labels)
    return np.hstack([np.vstack(result), labels[:, None]])


def get_distmap(a_result, b_result, mode="giou"):
    a_result = add_label(a_result)
    b_result = add_label(b_result)
    n = len(a_result)
    m = len(b_result)

    dist_iou = get_bbox_overlaps(a_result[:, :4], b_result[:, :4], mode=mode)
    if mode == "giou":  # giou range [-1, 1] -> [0, 1]
        dist_iou += 1
        dist_iou *= 0.5

    dist_iou = 1 - dist_iou

    dist_cls = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            if a_result[i, -1] == b_result[j, -1]:
                dist_cls[i, j] = a_result[i, -2] - b_result[j, -2]
            else:
                dist_cls[i, j] = 1

    dist_cls = np.abs(dist_cls)

    return (dist_iou + dist_cls) * 0.5


def get_ot_cost(a_detection, b_detection, return_matrix=False):
    """[summary]

    Args:
        a_detection (list): list of detection results. a_detection[i] contains bounding boxes for i-th class.
        Each element is numpy array whose shape is N x 5. [[x1, y1, x2, y2, s], ...]
        b_detection (list): ditto
    Returns:
        [float]: optimal transportation cost
    """
    is_a_none = a_detection is None
    is_b_none = b_detection is None
    if is_a_none and is_b_none:
        return 0  # no object is detected in both results
    if is_a_none or is_b_none:
        return 1  #

    M = get_distmap(a_detection, b_detection)
    n, m = M.shape

    return ot.emd2(
        np.ones(n) / n, np.ones(m) / m, M, return_matrix=return_matrix
    )
