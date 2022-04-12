import numpy as np
from src.extensions.dataset.coco_custom import CocoOtcDataset
from mmcv.utils.config import Config
import os
from mim.utils import DEFAULT_CACHE_DIR
from data.conf.model_cfg import MODEL_CFGS
import json
import matplotlib.pyplot as plt


def run():
    results = []
    model_cfg = "vfnet_r50_fpn_mstrain_2x_coco"
    cfg = Config.fromfile(os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"))
    dataset = CocoOtcDataset(
        "data/coco/annotations/instances_val2017.json",
        cfg.data.test.pipeline,
        test_mode=True,
    )
    for nl in np.arange(0, 0.7, 0.05):
        measures = dataset.evaluate_gt(nl)
        results.append(measures)

    for measure in ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "mOTC"]:
        vals = [x[measure] for x in results]
        plt.plot(vals, label=measure)
    plt.legend()
    plt.savefig("tmp/eval_gt_w_noise.pdf")
    json.dump(results, open("tmp/eval_gt_w_noise.json", "w"))


if __name__ == "__main__":
    run()
