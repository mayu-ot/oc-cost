from mmdet.core.evaluation import eval_map
import numpy as np

# https://github.com/open-mmlab/mmdetection/blob/6cf9aa1866b745fce8f1da6717fdb833d7c66fab/tools/analysis_tools/analyze_results.py


def bbox_map_eval(det_result, annotation):
    """Evaluate mAP of single image det result.
    Args:
        det_result (list[list]): [[cls1_det, cls2_det, ...], ...].
            The outer list indicates images, and the inner list indicates
            per-class detected bboxes.
        annotation (dict): Ground truth annotations where keys of
             annotations are:
            - bboxes: numpy array of shape (n, 4)
            - labels: numpy array of shape (n, )
            - bboxes_ignore (optional): numpy array of shape (k, 4)
            - labels_ignore (optional): numpy array of shape (k, )
    Returns:
        float: mAP
    """

    # use only bbox det result
    if isinstance(det_result, tuple):
        bbox_det_result = [det_result[0]]
    else:
        bbox_det_result = [det_result]
    # mAP
    iou_thrs = np.linspace(
        0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True
    )
    mean_aps = []
    results = []
    for thr in iou_thrs:
        mean_ap, result = eval_map(
            bbox_det_result, [annotation], iou_thr=thr, logger="silent"
        )
        mean_aps.append(mean_ap)
        results.append(result)
    return sum(mean_aps) / len(mean_aps), results
