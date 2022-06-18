from typing import List
from mim import test, download, get_model_info
from mim.utils import DEFAULT_CACHE_DIR
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import click
import time
from data.conf.model_cfg import MODEL_CFGS
from src.utils.neptune_utils import load_hparam_cfg

try:
    import neptune.new as neptune
    from neptune.new.types import File
except ImportError:
    raise ImportWarning("neptune client is not installed")


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


def generate_reports(
    out_dir: str,
    ncols: int,
    neptune_run_id: str,
    metrics: List[str] = ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "mOTC"],
):
    sns.set_style("white")
    work_dir = out_dir
    results = []
    files = os.listdir(work_dir)
    files.sort()
    for fn in files:
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

    nrows = len(metrics) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
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

    if neptune_run_id:
        proj_name = os.environ["NEPTUNE_PROJECT"]
        nptn_run = neptune.init(
            project=proj_name,
            run=neptune_run_id,
            capture_hardware_metrics=False,
        )
        nptn_run["evaluation/figs/summary"].upload(File.as_image(fig))
        nptn_run.stop()

    fig.savefig(os.path.join(work_dir, "metric.pdf"), bbox_inches="tight")


@cli.command()
@click.argument(
    "out_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.option("--ncols", type=int, default=4)
@click.option("--neptune-run-id", type=str, default="")
def generate_reports_cmd(
    out_dir: str,
    ncols: int,
    neptune_run_id: str,
    metrics: List[str] = ["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "mOTC"],
):
    return generate_reports(out_dir, ncols, neptune_run_id, metrics)


@cli.command()
@click.argument("out_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.argument("n_gpus", default=None)
@click.option("--neptune-on", is_flag=True)
@click.option("--use-tuned-hparam", default="")
@click.option("--alpha", default=0.5)
@click.option("--beta", default=0.6)
@click.option("--show-on", is_flag=True)
@click.option("--eval-options", type=str, multiple=True)
@click.option("-s", "--run-subset", is_flag=True)
def evaluate(
    out_dir,
    n_gpus,
    neptune_on,
    use_tuned_hparam,
    alpha,
    beta,
    show_on,
    eval_options,
    run_subset,
):
    """Evaluate OC-cost for detectors.

    Args:
        out_dir (str): output directory. All result files will be stored in a sub-directory with timestamp (out_dir/%Y%m%d_%H%M%S/).
        neptune_on (bool): When neptune_on is True, the experiment will be uploaded to the neptune server.
        use_tuned_hparam (str): download tuned parameters from neptune server. Set values either ["alpha=0.5,beta=0.6", "mAP"]
        alpha (float): OC-cost parameter. This parameter is called lambda in the paper.
        beta (float): _description_
        show_on (flag): write detection result images in output directory
        eval_options (_type_): _description_
        run_subset (flag): for test run. The evaluation is done on a small subset of validation set.
    """

    args = locals()

    tags = [f"alpha={alpha}", f"beta={beta}"]
    if len(use_tuned_hparam):
        tags.append("use-tuned-param")
    if run_subset:
        tags.append("run-subset")

    nptn_cfg = []
    nptn_run_id = ""
    if neptune_on:
        proj_name = os.environ["NEPTUNE_PROJECT"]
        run = neptune.init(
            proj_name,
            name="run_evaluation",
            mode="sync",
            capture_hardware_metrics=False,
            tags=tags,
        )
        nptn_run_id = run._short_id
        nptn_cfg = [
            f"data.test.nptn_project_id={proj_name}",
            f"data.test.nptn_run_id={nptn_run_id}",
            "",
        ]
        run["params"] = args
    if run_subset:
        data_cfg = [
            "data.test.ann_file=data/coco/annotations/instances_val2017_subset.json"
        ]
    else:
        data_cfg = []

    eval_options = (
        ["--eval-options"]
        + [f"otc_params=[(alpha, {alpha}), (beta, {beta})]"]
        + [x for x in eval_options]
    )

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

        # test hyperparameters
        hparam_options = ()
        if len(use_tuned_hparam):
            hparam_options = load_hparam_cfg(model_cfg, use_tuned_hparam)

        if len(nptn_cfg):
            nptn_cfg[
                -1
            ] = f"data.test.nptn_metadata_suffix={MODEL_CFGS[model_cfg]}"

        out_pkl = f"{os.path.join(out_dir, MODEL_CFGS[model_cfg]+'.pkl')}"
        other_args = [
            "--out",
            out_pkl,
            "--eval",
            "bbox",
            "--work-dir",
            f"{out_dir}",
            "--cfg-options",
            f"data.test.type=CocoOtcDataset",
            *nptn_cfg,
            *data_cfg,  # to run evaluation on a small subset
            "custom_imports.imports=[src.extensions.dataset.coco_custom, src.utils.matplotlib_settings]",
            "custom_imports.imports=[src.extensions.dataset.coco_custom]",
            "custom_imports.allow_failed_imports=False",
            *hparam_options,
            *eval_options,
        ]
        if show_on:
            show_dir = os.path.join(out_dir, MODEL_CFGS[model_cfg])
            if not os.path.exists(show_dir):
                os.makedirs(show_dir)
            other_args = [
                "--show",
                "--show-score-thr",
                "0.0",
                "--show-dir",
                show_dir,
            ] + other_args

            _ = test(
                package="mmdet",
                config=os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"),
                checkpoint=os.path.join(DEFAULT_CACHE_DIR, checkpoint_name),
                other_args=other_args,
            )
        else:
            _ = test(
                package="mmdet",
                config=os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"),
                checkpoint=os.path.join(DEFAULT_CACHE_DIR, checkpoint_name),
                gpus=n_gpus,
                launcher="pytorch",
                other_args=other_args,
            )

        if neptune_on:
            run[f"other_args/{MODEL_CFGS[model_cfg]}"] = json.dumps(other_args)
            run[f"results/{MODEL_CFGS[model_cfg]}"].upload(out_pkl)

    generate_reports(out_dir, 4, nptn_run_id)

    if neptune_on:
        run.stop()


if __name__ == "__main__":
    cli()
