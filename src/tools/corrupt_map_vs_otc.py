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

from src.utils.neptune_utils import load_hparam_neptune
from data.conf.model_cfg import MODEL_CFGS
import pdb


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
def generate_reports(
    out_dir, measures=["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "mOTC"]
):
    sns.set_style("white")
    work_dir = out_dir

    results = {}
    for data_type in os.listdir(work_dir):
        if not os.path.isdir(os.path.join(work_dir, data_type)):
            continue
        for fn in os.listdir(os.path.join(work_dir, data_type)):
            if fn.endswith(".json"):
                res = json.load(open(os.path.join(work_dir, data_type, fn)))
                cfg_name = os.path.splitext(os.path.basename(res["config"]))[0]
                model_name = MODEL_CFGS[cfg_name]
                results.setdefault(model_name, {})
                results[model_name][data_type] = res

    f, axes = plt.subplots(
        len(measures),
        len(results),
        figsize=(5 * len(results), 5 * len(measures)),
        sharey=True,
    )
    for i, measure in enumerate(measures):
        axes[i][0].set_ylabel(measure)

        for j, (k, res_dict) in enumerate(results.items()):
            ax = axes[i][j]
            ax.set_title(k)
            labels = list(res_dict.keys())
            scores = [res_dict[l]["metric"][measure] for l in labels]
            ax.bar(range(len(labels)), scores)
            ax.set_xticks(range(len(labels)))
            ax.set_xticklabels(labels, rotation=45)

    f.savefig(os.path.join(work_dir, "metric.pdf"), bbox_inches="tight")


@cli.command()
@click.argument("dataset")
@click.argument("out_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--use-tuned-hparam", default="")
@click.option("--alpha", default=0.5)
@click.option("--beta", default=0.5)
@click.option("-s", "--run-subset", is_flag=True)
def evaluate(dataset, out_dir, use_tuned_hparam, alpha, beta, run_subset):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = os.path.join(out_dir, timestamp)
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    log_file = os.path.join(out_dir, "corrupt_map_vs_otc.log")
    logger = get_root_logger(log_file)
    logger.info(f"dataset={dataset}")
    logger.info(f"dataset={out_dir}")

    eval_options = ["--eval-options"] + [
        f"otc_params=[(alpha, {alpha}), (beta, {beta}), (use_dummy, True)]"
    ]

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
            hparam_options = ()
            if len(use_tuned_hparam):
                hparam_options = load_hparam_neptune(
                    model_cfg, use_tuned_hparam
                )

            if run_subset:
                data_cfg = [
                    "data.test.ann_file=data/coco/annotations/instances_val2017_subset.json"
                ]
            else:
                data_cfg = []

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
                    *data_cfg,
                    "custom_imports.imports=[src.extensions.dataset.coco_custom]",
                    "custom_imports.allow_failed_imports=False",
                    *hparam_options,
                    *eval_options,
                ),
            )


if __name__ == "__main__":
    cli()
