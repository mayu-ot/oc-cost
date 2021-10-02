from mim import test, download, get_model_info
from mim.utils import DEFAULT_CACHE_DIR
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import click
import time

MODEL_CFGS = {
    "retinanet_r50_fpn_2x_coco": "RetinaNet",
    "faster_rcnn_r50_fpn_2x_coco": "Faster-RCNN",
    "yolof_r50_c5_8x8_1x_coco": "YOLOF",
    "detr_r50_8x2_150e_coco": "DETR",
    "vfnet_r50_fpn_mstrain_2x_coco": "VFNet",
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
@click.option("--ncols", type=int, default=3)
def generate_reports(
    out_dir, ncols, metrics=["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "mOTC"]
):
    sns.set_style("white")
    work_dir = out_dir
    results = []
    for fn in os.listdir(work_dir):
        if os.path.splitext(fn)[-1] == ".json":
            res = json.load(open(os.path.join(work_dir, fn)))
            results.append(res)

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

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_infos = get_model_info("mmdet")

    for model_cfg in MODEL_CFGS.keys():

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
            gpus=4,
            launcher="pytorch",
            other_args=(
                "--eval",
                "bbox",
                "--work-dir",
                f"{out_dir}",
                "--cfg-options",
                f"data.test.type={dataset}",
                # "data.test.ann_file=data/coco/annotations/instances_val2017_subset.json",  # to run evaluation on a small subset
                "custom_imports.imports=[src.extensions.dataset.coco_custom]",
                "custom_imports.allow_failed_imports=False",
            ),
        )


if __name__ == "__main__":
    cli()
