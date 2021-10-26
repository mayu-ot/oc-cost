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


def cost_func(x, y, mode: str = "giou", alpha: float = 0.8):
    loc_cost = (
        1 - get_bbox_overlaps(x[:4][None, :], y[:4][None, :], mode)
    ) * 0.5  # normalized to [0, 1]
    l_x, l_y = x[-1], y[-1]
    if l_x == l_y:
        cls_cost = np.abs(x[-2] - y[-2])
    else:
        cls_cost = x[-2] + y[-2]
    cls_cost *= 0.5  # normalized to [0, 1]

    return alpha * loc_cost + (1 - alpha) * cls_cost


def get_cmap(
    a_result, b_result, alpha=0.8, beta=0.4, mode="giou", use_dummy=True
):
    """[summary]

    Args:
        a_result ([type]): ground truth bounding boxes.
        b_result ([type]): predictions
        mode (str, optional): [description]. Defaults to "giou".

    Returns:
        dist_a (np.array): (N+1,) array. distribution over ground truth bounding boxes.
        dist_b (np.array): (M,) array. distribution over predictions.
        cost_map:
    """
    a_result = add_label(a_result)
    b_result = add_label(b_result)
    n = len(a_result)
    m = len(b_result)

    cost_map = np.zeros((n + int(use_dummy), m + int(use_dummy)))

    metric = lambda x, y: cost_func(x, y, alpha=alpha, mode=mode)
    cost_map[:n, :m] = ot.utils.dist(a_result, b_result, metric)

    dist_a = np.ones(n + int(use_dummy))
    dist_b = np.ones(m + int(use_dummy))

    # cost for dummy demander / supplier
    if use_dummy:
        cost_map[-1, :] = beta
        cost_map[:, -1] = beta
        dist_a[-1] = m
        dist_b[-1] = n

    dist_a /= dist_a.sum()
    dist_b /= dist_b.sum()

    return dist_a, dist_b, cost_map


def subtract_dummy2dummy_cost(M, outputs):
    if len(outputs) == 2:
        total_cost, log = outputs
    G = log["G"]
    return total_cost - M[-1, -1] * G[-1, -1]


def get_ot_cost(a_detection, b_detection, cmap_func, return_matrix=False):
    """[summary]

    Args:
        a_detection (list): list of detection results. a_detection[i] contains bounding boxes for i-th class.
        Each element is numpy array whose shape is N x 5. [[x1, y1, x2, y2, s], ...]
        b_detection (list): ditto
        cmap_func (callable): a function that takes a_detection and b_detection as input and returns a unit cost matrix
    Returns:
        [float]: optimal transportation cost
    """
    is_a_none = a_detection is None
    is_b_none = b_detection is None

    if not is_a_none:
        if sum([len(x) for x in a_detection]) == 0:
            is_a_none = True

    if not is_b_none:
        if sum([len(x) for x in b_detection]) == 0:
            is_b_none = True

    if is_a_none and is_b_none:
        return 0  # no object is detected in both results
    if is_a_none or is_b_none:
        return 1  #

    a, b, M = cmap_func(a_detection, b_detection)
    outputs = ot.emd2(a, b, M, return_matrix=return_matrix)

    return outputs
