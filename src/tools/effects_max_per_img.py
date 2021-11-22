from mim import test, download, get_model_info
from mim.utils import DEFAULT_CACHE_DIR
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import click
import time
import neptune.new as neptune
from neptune.new.types import File
from data.conf.model_cfg import MODEL_CFGS
import glob
import tempfile


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
    res_files = os.listdir(work_dir)
    res_files.sort()

    print(res_files)

    results = {}
    for fn in res_files:
        if os.path.splitext(fn)[-1] == ".json":
            res = json.load(open(os.path.join(work_dir, fn)))
            for k, v in res["metric"].items():
                results.setdefault(k, []).append(v)

    nrows = len(metrics) // ncols
    f, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    axes = axes.ravel()

    for i, metric in enumerate(metrics):

        bars = axes[i].bar(
            range(20, 120, 20),
            results[metric],
            width=15,
        )
        axes[i].set_ylim(0, 1.0)
        axes[i].set_title(metric)
        stylize_bars(bars, axes[i], "k")

    f.savefig(
        os.path.join(work_dir, "effects_max_per_img.pdf"), bbox_inches="tight"
    )


@cli.command()
@click.argument("run_id", type=str)
def upload_reports(run_id):
    run = neptune.init(project=os.environ["NEPTUNE_PROJECT"], run=run_id)
    measures = {}
    with tempfile.TemporaryDirectory() as tmpdir:
        for m in range(10, 200, 20):
            run[f"measures/max_per_img={m}"].download(tmpdir)
            measures[m] = json.load(
                open(os.path.join(tmpdir, f"max_per_img={m}.json"))
            )

    sns.set("paper")
    x = []
    mAP = []
    otc = []
    for k, v in measures.items():
        x.append(k)
        mAP.append(v["metric"]["bbox_mAP"])
        otc.append(v["metric"]["mOTC"])

    f, axes = plt.subplots(1, 2, sharex=True, figsize=(10, 3))
    axes[0].plot(x, mAP)
    axes[0].set_title("mAP vs detections per image")

    axes[1].plot(x, otc)
    axes[1].set_title("mOTC vs detections per image")

    run["figs/performance_vs_det_per_img"].upload(File.as_image(f))

    f_name = os.path.join("tmp/", "performance_vs_det_per_img.pdf")
    f.savefig(f_name, bbox_inches="tight")
    run["figs/performance_vs_det_per_img_pdf"].upload(f_name)
    run.stop()


@cli.command()
@click.argument("dataset")
@click.argument("out_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--neptune-on", is_flag=True)
@click.option("-s", "--run-subset", is_flag=True)
def evaluate(dataset, out_dir, neptune_on, run_subset):
    args = locals()
    if neptune_on:
        proj_name = os.environ["NEPTUNE_PROJECT"]
        run = neptune.init(
            proj_name,
            name="effects_max_per_img",
            mode="sync",
            capture_hardware_metrics=False,
            tags=["behavior analysis"],
        )
        run["params"] = args

    if run_subset:
        data_cfg = [
            "data.test.ann_file=data/coco/annotations/instances_val2017_subset.json"
        ]
    else:
        data_cfg = []

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = os.path.join(out_dir, timestamp)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_infos = get_model_info("mmdet")
    model_cfg = "vfnet_r50_fpn_mstrain_2x_coco"

    if not os.path.exists(os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py")):
        download("mmdet", [model_cfg])

    model_info = model_infos.loc[model_cfg]
    checkpoint_name = os.path.basename(model_info.weight)

    for m in range(10, 200, 20):  # range(20, 120, 20):
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
                f"{out_dir}",
                "--cfg-options",
                f"data.test.type={dataset}",
                *data_cfg,
                f"model.test_cfg.max_per_img={m}",
                "custom_imports.imports=[src.extensions.dataset.coco_custom]",
                "custom_imports.allow_failed_imports=False",
            ),
        )

        files = glob.glob(os.path.join(out_dir, "*.json"))
        latest_file = max(files, key=os.path.getctime)
        if neptune_on:
            run[f"measures/max_per_img={m}"].upload(latest_file)


if __name__ == "__main__":
    cli()
