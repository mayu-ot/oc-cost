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
import neptune.new as neptune
from neptune.new.types import File
from data.conf.model_cfg import MODEL_CFGS
from src.tools.human_consistency import prepare_data
from src.utils.neptune_utils import load_hparam_neptune
from mmcv import Config
from mmdet.datasets import build_dataset, get_loading_pipeline
from src.extensions.dataset.coco_custom import CocoOtcDataset
from src.utils.neptune_utils import load_results
from concurrent.futures import ProcessPoolExecutor, as_completed
import pandas as pd
import seaborn as sns
from tqdm import tqdm


@click.group()
def cli():
    pass


def prepare_dataset(model_cfg):
    cfg = Config.fromfile(os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"))
    cfg.data.test.type = "CocoOtcDataset"
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    return dataset


def _eval(results, alpha, beta):
    dataset = prepare_dataset(
        "vfnet_r50_fpn_mstrain_2x_coco",
    )
    ot_cost = dataset.eval_OTC(results, alpha=alpha, beta=beta)
    return alpha, beta, ot_cost


def plot_ranking(ranking, param_name, out_dir):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 3))
    for model_name, row in ranking.iterrows():
        ranks = row.values
        plt.plot(ranking.columns.to_numpy(), ranks, label=model_name)
    plt.xlabel(param_name)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, f"{param_name}_analysis_ranking.pdf"),
        bbox_inches="tight",
    )
    plt.close("all")


def plot_costs(costs, param_name, out_dir):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(10, 3))
    for model_name, row in costs.iterrows():
        vals = row.values
        plt.plot(costs.columns.to_numpy(), vals, label=model_name)
    plt.xlabel(param_name)
    plt.legend()
    plt.savefig(
        os.path.join(out_dir, f"{param_name}_analysis_costs.pdf"),
        bbox_inches="tight",
    )
    plt.close("all")


@cli.command()
@click.argument("run_id")
@click.argument("out_dir", type=click.Path(file_okay=False, dir_okay=True))
@click.option("--neptune-on", is_flag=True)
def run(
    run_id,
    out_dir,
    neptune_on,
):
    args = locals()

    if neptune_on:
        proj_name = os.environ["NEPTUNE_PROJECT"]
        run = neptune.init(
            proj_name,
            name="ranking_w_diff_params",
            mode="sync",
            capture_hardware_metrics=False,
        )
        run["params"] = args

    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = os.path.join(out_dir, timestamp)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_names = MODEL_CFGS.values()
    alphas = np.arange(0.0, 1.1, 0.1)
    betas = np.arange(0.1, 1.0, 0.1)
    alpha_results = pd.DataFrame(index=model_names, columns=alphas)
    beta_results = pd.DataFrame(index=model_names, columns=betas)

    for model_name in MODEL_CFGS.values():
        detections = load_results(run_id, model_name)
        beta = 0.6

        progress = tqdm(total=len(alphas))

        with ProcessPoolExecutor(8) as pool:
            futures = []

            for alpha in alphas:
                future = pool.submit(_eval, detections, alpha, beta)
                futures.append(future)

            for future in as_completed(futures):
                progress.update(1)
                alpha, beta, ot_cost = future.result()
                alpha_results.at[model_name, alpha] = ot_cost

        alpha = 0.5
        progress = tqdm(total=len(betas))

        with ProcessPoolExecutor(8) as pool:
            futures = []

            for beta in betas:
                future = pool.submit(_eval, detections, alpha, beta)
                futures.append(future)

            for future in as_completed(futures):
                progress.update(1)
                alpha, beta, ot_cost = future.result()
                beta_results.at[model_name, beta] = ot_cost

    alpha_ranking = alpha_results.rank(axis=0)
    alpha_results.to_csv(os.path.join(out_dir, "lambda_ot_costs.csv"))
    alpha_ranking.to_csv(os.path.join(out_dir, "lambda_ranking.csv"))

    plot_ranking(alpha_ranking, "lambda", out_dir)
    plot_costs(alpha_results, "lambda", out_dir)

    beta_ranking = beta_results.rank(axis=0)
    beta_results.to_csv(os.path.join(out_dir, "beta_ot_costs.csv"))
    beta_ranking.to_csv(os.path.join(out_dir, "beta_ranking.csv"))

    plot_ranking(beta_ranking, "beta", out_dir)
    plot_costs(beta_results, "beta", out_dir)

    if neptune_on:
        run.stop()


if __name__ == "__main__":
    cli()
