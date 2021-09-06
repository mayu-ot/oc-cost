from mmdet.core.evaluation.bbox_overlaps import bbox_overlaps
import ot
import numpy as np
import pdb


def add_label(result):
    labels = [[i] * len(r) for i, r in enumerate(result)]
    labels = np.hstack(labels)
    return np.hstack([np.vstack(result), labels[:, None]])


def get_distmap(a_result, b_result):
    a_result = add_label(a_result)
    b_result = add_label(b_result)
    n = len(a_result)
    m = len(b_result)

    dist_iou = 1 - bbox_overlaps(a_result[:, :4], b_result[:, :4])
    dist_cls = np.zeros((n, m))

    for i in range(n):
        for j in range(m):
            if a_result[i, -1] == b_result[j, -1]:
                dist_cls[i, j] = a_result[i, -2] - b_result[j, -2]
            else:
                dist_cls[i, j] = 1

    dist_cls = np.abs(dist_cls)

    return (dist_iou + dist_cls) * 0.5


def get_ot_cost(a_detection, b_detection):
    is_a_none = a_detection is None
    is_b_none = b_detection is None
    if is_a_none and is_b_none:
        return 0  # no object is detected in both results
    if is_a_none or is_b_none:
        return 1  #

    M = get_distmap(a_detection, b_detection)
    n, m = M.shape

    c = ot.emd2(np.ones(n) / n, np.ones(m) / m, M)
    return c
