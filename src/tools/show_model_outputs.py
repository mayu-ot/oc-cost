from mim import test, download, get_model_info
from mim.utils import DEFAULT_CACHE_DIR
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pdb
import click

MODEL_CFGS = {
    # "retinanet_r50_fpn_2x_coco": "RetinaNet",
    "faster_rcnn_r50_fpn_2x_coco": "Faster-RCNN",
    # "yolof_r50_c5_8x8_1x_coco": "YOLOF",
    # "detr_r50_8x2_150e_coco": "DETR",
    # "vfnet_r50_fpn_mstrain_2x_coco": "VFNet",
}


def run_test():
    model_infos = get_model_info("mmdet")
    for model_cfg in MODEL_CFGS.keys():
        model_name = MODEL_CFGS[model_cfg]
        if not os.path.exists(
            os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py")
        ):
            download("mmdet", [model_cfg])

        model_info = model_infos.loc[model_cfg]
        checkpoint_name = os.path.basename(model_info.weight)

        _ = test(
            package="mmdet",
            config=os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"),
            checkpoint=os.path.join(DEFAULT_CACHE_DIR, checkpoint_name),
            other_args=(
                "--eval",
                "bbox",
                "--show",
                "--show-score-thr",
                "0.0",
                "--show-dir",
                f"outputs/shows/{model_name}/",
                "--out",
                f"tmp/{model_name}.pkl",
                "--cfg-options",
                "data.test.type=CocoOtcDataset",
                "data.test.ann_file=data/coco/annotations/instances_val2017_subset.json",
                "custom_imports.imports=[src.extensions.dataset.coco_custom]",
                "custom_imports.allow_failed_imports=False",
            ),
        )


if __name__ == "__main__":
    run_test()
