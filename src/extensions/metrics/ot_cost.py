from typing import Callable, Sequence, Tuple, Callable, Union
from jupyter_bbox_widget import bbox
from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import ot
import numpy as np
import numpy.typing as npt


def get_bbox_overlaps(
    bboxes1: npt.ArrayLike, bboxes2: npt.ArrayLike, mode: str = "iou", eps: float = 1e-6
):
    if mode == "iou":
        return bbox_overlaps(bboxes1, bboxes2, eps=eps)
    elif mode == "giou":
        return bbox_gious(bboxes1, bboxes2, eps=eps)
    else:
        raise RuntimeError("valid mode is iou or giou")


def bbox_gious(
    bboxes1: npt.ArrayLike,
    bboxes2: npt.ArrayLike,
    eps: float = 1e-6,
    use_legacy_coordinate: bool = False,
) -> npt.ArrayLike:
    """Calculate the generalized ious between each bbox of bboxes1 and bboxes2.
    Args:
        bboxes1 (ndarray): Shape (n, 4) # [[x1, y1, x2, y2], ...]
        bboxes2 (ndarray): Shape (k, 4) # [[x1, y1, x2, y2], ...]
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


def add_label(result: Sequence[Sequence]) -> npt.ArrayLike:
    labels = [[i] * len(r) for i, r in enumerate(result)]
    labels = np.hstack(labels)
    return np.hstack([np.vstack(result), labels[:, None]])


def cost_func(x, y, mode: str = "giou", alpha: float = 0.8):
    """Calculate a unit cost

    Args:
        x (np.ndarray): a detection [x1, y1, x2, y2, s, l]. s is a confidence value, and l is a classification label.
        y (np.ndarray): a detection [x1, y1, x2, y2, s, l]. s is a confidence value, and l is a classification label.
        mode (str, optional): Type of IoUs. Defaults to "giou" (Generalized IoU).
        alpha (float, optional): weights to balance localization and classification errors. Defaults to 0.8.

    Returns:
        float: a unit cost
    """
    giou_val = get_bbox_overlaps(x[:4][None, :], y[:4][None, :], mode)  # range [-1, 1]
    loc_cost = 1 - (giou_val + 1) * 0.5  # normalized to [0, 1]
    l_x, l_y = x[-1], y[-1]
    if l_x == l_y:
        cls_cost = np.abs(x[-2] - y[-2])
    else:
        cls_cost = x[-2] + y[-2]
    cls_cost *= 0.5  # normalized to [0, 1]

    return alpha * loc_cost + (1 - alpha) * cls_cost


def get_cmap(
    a_result: Sequence[npt.ArrayLike],
    b_result: Sequence[npt.ArrayLike],
    alpha: float = 0.8,
    beta: float = 0.4,
    mode="giou",
) -> Tuple[npt.ArrayLike]:
    """Calculate cost matrix

    Args:
        a_result ([type]): detections
        b_result ([type]): detections
        mode (str, optional): [description]. Defaults to "giou".

    Returns:
        dist_a (np.array): (N+1,) array. distribution over detections.
        dist_b (np.array): (M+1,) array. distribution over detections.
        cost_map:
    """
    a_result = add_label(a_result)
    b_result = add_label(b_result)
    n = len(a_result)
    m = len(b_result)

    cost_map = np.zeros((n + 1, m + 1))

    metric = lambda x, y: cost_func(x, y, alpha=alpha, mode=mode)
    cost_map[:n, :m] = ot.utils.dist(a_result, b_result, metric)

    dist_a = np.ones(n + 1)
    dist_b = np.ones(m + 1)

    # cost for dummy demander / supplier
    cost_map[-1, :] = beta
    cost_map[:, -1] = beta
    dist_a[-1] = m
    dist_b[-1] = n

    return dist_a, dist_b, cost_map


def postprocess(M: npt.ArrayLike, total_cost: float, log: dict) -> float:
    G = log["G"]
    G[-1, -1] = 0
    G /= G.sum()
    total_cost = (M * G).sum()
    return total_cost


def get_ot_cost(
    a_detection: list,
    b_detection: list,
    cmap_func: Callable,
    return_matrix: bool = False,
) -> Union[float, Tuple[float, dict]]:
    """[summary]

    Args:
        a_detection (list): list of detection results. a_detection[i] contains bounding boxes for i-th class.
        Each element is numpy array whose shape is N x 5. [[x1, y1, x2, y2, s], ...]
        b_detection (list): ditto
        cmap_func (callable): a function that takes a_detection and b_detection as input and returns a unit cost matrix
    Returns:
        [float]: optimal transportation cost
    """

    if sum(map(len, a_detection)) == 0:
        if sum(map(len, b_detection)) == 0:
            return 0

    a, b, M = cmap_func(a_detection, b_detection)
    total_cost, log = ot.emd2(a, b, M, return_matrix=True)
    total_cost = postprocess(M, total_cost, log)

    if return_matrix:
        log["M"] = M
        log["a"] = a
        log["b"] = b
        return total_cost, log
    else:
        return total_cost
