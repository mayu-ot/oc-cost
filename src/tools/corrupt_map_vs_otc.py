from mim import test, download, get_model_info
from mim.utils import DEFAULT_CACHE_DIR
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import click
import time
from mmdet.utils.logger import get_root_logger

MODEL_CFGS = {
    "retinanet_r50_fpn_2x_coco": "RetinaNet",
    "faster_rcnn_r50_fpn_2x_coco": "Faster-RCNN",
}


@click.group()
def cli():
    pass


def stylize_bars(bars, ax, txt_color="w"):
    ranks = np.argsort([b.get_height() for b in bars])
    cmap = sns.light_palette("seagreen", reverse=True)
    for _, bar in enumerate(bars):
        ax.text(
            bar.get_x() + bar.get_width() * 0.5,
            bar.get_height() - 0.05,
            f"{bar.get_height():.2f}",
            horizontalalignment="center",
            fontsize=9,
            color=txt_color,
        )

    for r in ranks:
        bars[r].set_color(cmap.pop())


@cli.command()
@click.argument(
    "out_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option("--ncols", type=int, default=4)
def generate_reports(
    out_dir, ncols, metrics=["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "mOTC"]
):
    sns.set_style("white")
    work_dir = out_dir

    results = {}
    for data_type in ["corrupt", "org"]:
        for fn in os.listdir(os.path.join(work_dir, data_type)):
            if fn.endswith(".json"):
                res = json.load(open(os.path.join(work_dir, fn)))
                cfg_name = os.path.splitext(os.path.basename(res["config"]))[0]
                model_name = MODEL_CFGS[cfg_name]
                results[model_name] = (data_type, res)

    models = []
    for res in results:
        cfg_name = os.path.splitext(os.path.basename(res["config"]))[0]
        models.append(MODEL_CFGS[cfg_name])

    data = {}
    data["model"] = models

    for metric in metrics:
        vals = [res["metric"][metric] for res in results]
        data[metric] = vals

    nrows = len(metrics) // ncols + 1
    f, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.ravel()

    for i, metric in enumerate(metrics):

        bars = axes[i].bar(
            np.arange(len(models)),
            data[metric],
            width=0.3,
        )
        axes[i].set_ylim(0, 1.0)
        axes[i].set_title(metric)
        stylize_bars(bars, axes[i], "k")

        axes[i].set_xticks(np.arange(len(models)))
        axes[i].set_xticklabels(models, rotation=45)

    f.savefig(os.path.join(work_dir, "metric.pdf"), bbox_inches="tight")


@cli.command()
@click.argument("dataset")
@click.argument("out_dir", type=click.Path(file_okay=False, dir_okay=True))
def evaluate(dataset, out_dir):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = os.path.join(out_dir, timestamp)
    log_file = os.path.join(out_dir, "corrupt_map_vs_otc.log")
    logger = get_root_logger(log_file)
    logger.info(f"dataset={dataset}")
    logger.info(f"dataset={out_dir}")

    for data_type in ["GaussNoise", "ImageCompression", "org"]:
        out_sub_dir = os.path.join(out_dir, data_type)
        if data_type == "org":
            img_dir = "data/coco/val2017/"
        else:
            img_dir = f"data/processed/coco-corrupted/val2017/{data_type}/"

        if not os.path.exists(out_sub_dir):
            os.makedirs(out_sub_dir)

        model_infos = get_model_info("mmdet")

        for model_cfg in MODEL_CFGS.keys():

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

            _ = test(
                package="mmdet",
                config=os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"),
                checkpoint=os.path.join(DEFAULT_CACHE_DIR, checkpoint_name),
                gpus=2,
                launcher="pytorch",
                other_args=(
                    "--eval",
                    "bbox",
                    "--work-dir",
                    f"{out_sub_dir}",
                    "--cfg-options",
                    f"data.test.type={dataset}",
                    f"data.test.img_prefix={img_dir}",
                    "data.test.ann_file=data/coco/annotations/instances_val2017.json",  # to run evaluation on a small subset
                    "custom_imports.imports=[src.extensions.dataset.coco_custom]",
                    "custom_imports.allow_failed_imports=False",
                    f"model.test_cfg.score_thr={score_thr}",
                    f"model.test_cfg.nms.iou_threshold={iou_threshold}",
                ),
            )


if __name__ == "__main__":
    cli()
