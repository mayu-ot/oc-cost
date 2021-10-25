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
    "retinanet_r50_fpn_2x_coco": "RetinaNet",
    # "faster_rcnn_r50_fpn_2x_coco": "Faster-RCNN",
    # "yolof_r50_c5_8x8_1x_coco": "YOLOF",
    # "detr_r50_8x2_150e_coco": "DETR",
    # "vfnet_r50_fpn_mstrain_2x_coco": "VFNet",
}


@click.command()
@click.argument(
    "out_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
def run_test(out_dir):
    model_infos = get_model_info("mmdet")
    for model_cfg in MODEL_CFGS.keys():
        model_name = MODEL_CFGS[model_cfg]
        if not os.path.exists(
            os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py")
        ):
            download("mmdet", [model_cfg])

        model_info = model_infos.loc[model_cfg]
        checkpoint_name = os.path.basename(model_info.weight)

        # test hyperparameters
        hparams = json.load(
            open(
                f"data/processed/tune_hparams_otc/{MODEL_CFGS[model_cfg]}_tune_res.json"
            )
        )
        score_thr = hparams["best_params"]["score_thr"]
        iou_threshold = hparams["best_params"]["iou_threshold"]

        # img_dir = "data/processed/coco-corrupted/val2017/ImageCompression/"

        _ = test(
            package="mmdet",
            config=os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"),
            checkpoint=os.path.join(DEFAULT_CACHE_DIR, checkpoint_name),
            other_args=(
                "--eval",
                "bbox",
                "--show",
                "--show-score-thr",
                f"{score_thr}",
                "--show-dir",
                out_dir,
                "--out",
                f"tmp/{model_name}.pkl",
                "--cfg-options",
                "data.test.type=CocoOtcDataset",
                # f"data.test.img_prefix={img_dir}",
                "data.test.ann_file=data/coco/annotations/instances_val2017_subset.json",
                "custom_imports.imports=[src.extensions.dataset.coco_custom]",
                "custom_imports.allow_failed_imports=False",
                f"model.test_cfg.score_thr={score_thr}",
                f"model.test_cfg.nms.iou_threshold={iou_threshold}",
            ),
        )


if __name__ == "__main__":
    run_test()
