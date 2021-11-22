from mim import test, download, get_model_info
from mim.utils import DEFAULT_CACHE_DIR
import os
import json
import mmcv
from mmcv.utils.config import Config
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import click
import time
import pdb
import tempfile

from src.extensions.dataset.coco_custom import CocoOtcDataset
from src.utils.neptune_utils import load_results
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
import pandas as pd
from tabulate import tabulate
import neptune.new as neptune
from neptune.new.types import File
from data.conf.model_cfg import MODEL_CFGS
from src.utils.neptune_utils import load_hparam_neptune
from mmdet.datasets import build_dataset, get_loading_pipeline
from random import choices


@click.group()
def cli():
    pass


def get_overall_measures(out_dir):
    coco = CocoOtcDataset(
        "data/coco/annotations/instances_val2017.json",
        [],
        test_mode=True,
    )
    measures = {}
    for _, model_name in MODEL_CFGS.items():
        results = mmcv.load(os.path.join(out_dir, f"{model_name}.pkl"))
        measures[model_name] = coco.evaluate(results)
    return measures


@cli.command()
@click.argument(
    "out_dir", type=click.Path(exists=True, file_okay=False, dir_okay=True)
)
@click.argument("ratio", default=0.8)
def generate_reports_cmd(
    out_dir, ratio, measures=["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "mOTC"]
):
    return generate_reports(out_dir, ratio, measures)


def generate_reports(
    out_dir,
    ratio,
    measures=["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "mOTC"],
    nptn_run=None,
):
    sns.set_style("white")
    work_dir = out_dir
    files = glob.glob(os.path.join(work_dir, f"*.{ratio}.measures.json"))

    ncols = len(measures)

    f, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 6))

    data_frame = {measure: [] for measure in measures}
    data_frame["model"] = []

    for file in files:
        print(file)
        data = json.load(open(file))
        model_name = os.path.basename(file).split(".")[0]
        data_frame["model"] += [model_name] * len(data)
        for measure in measures:
            data_frame[measure] += [x[measure] for x in data]

    data_frame = pd.DataFrame(data_frame)
    for i, measure in enumerate(measures):
        ax = axes[i]
        sns.violinplot(x="model", y=measure, ax=ax, data=data_frame)
        ax.set_title(measure)
        y_min, _ = ax.get_ylim()
        ax.set_ylim(y_min, y_min + 0.15)
        ax.set_ylabel("")
        ax.set_xlabel("")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45)

    if nptn_run is not None:
        nptn_run["figs/summary"].upload(File.as_image(f))

    f.savefig(
        os.path.join(work_dir, f"{ratio}_measures_dist.pdf"),
        bbox_inches="tight",
    )

    display_bias(out_dir, measures, data_frame, ratio)


def display_bias(out_dir, measures, data_frame, ratio):
    measures_overall_file = os.path.join(out_dir, "all_measure.json")
    if os.path.exists(measures_overall_file):
        measures_overall = json.load(open(measures_overall_file))
    else:
        measures_overall = get_overall_measures(out_dir)
        json.dump(measures_overall, open(measures_overall_file, "w"))

    bias_table = []
    for model_name in MODEL_CFGS.values():
        entry = []
        for measure in measures:
            all_score = measures_overall[model_name][measure]
            sub_mean = data_frame[data_frame.model == model_name][
                measure
            ].mean()
            entry.append(f"{all_score:.4f}")
            entry.append(f"{sub_mean:.4f} ({sub_mean-all_score:.5f})")
        bias_table.append(entry)

    columns = np.ravel(
        [[measure, f"{measure} ({ratio*100}%)"] for measure in measures]
    )
    columns = columns.tolist()
    table = pd.DataFrame(np.asarray(bias_table), columns=columns)
    table["model"] = MODEL_CFGS.values()
    table.set_index("model")
    table_str = tabulate(table, headers="keys", tablefmt="psql")
    print(table_str)
    table.to_csv(os.path.join(out_dir, f"{ratio}_bias.csv"))


def prepare_dataset(model_cfg, data_file):
    cfg = Config.fromfile(os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"))
    cfg.data.test.type = "CocoOtcDataset"
    cfg.data.test.ann_file = data_file
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)
    return dataset


def eval_on_subset(out_pkl, model_cfg, ratio=0.8):
    results = mmcv.load(out_pkl)
    # write subset json file to tmp file
    dataset = json.load(
        open("data/coco/annotations/instances_val2017.json", "r")
    )
    n_dataset = len(dataset["images"])
    n_sub = int(n_dataset * ratio)

    sub_idx = choices(range(n_dataset), k=n_sub)
    new_ids = range(len(sub_idx))

    sub_results = [results[i] for i in sub_idx]

    sub_dataset = {
        "images": [],
        "annotations": [],
        "categories": dataset["categories"].copy(),
    }
    for i, n_id in zip(sub_idx, new_ids):
        sub_dataset["images"].append(dataset["images"][i].copy())
        c_id = sub_dataset["images"][-1]["id"]
        sub_dataset["images"][-1]["id"] = n_id
        anns = [
            x.copy() for x in dataset["annotations"] if x["image_id"] == c_id
        ]
        for ann in anns:
            ann["image_id"] = n_id
            ann["id"] = len(sub_dataset["annotations"])
            sub_dataset["annotations"].append(ann)

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = os.path.join(tmp_dir.name, "sub_dataset.json")
    json.dump(sub_dataset, open(tmp_file, "w"))

    dataset = prepare_dataset(model_cfg, tmp_file)
    measure = dataset.evaluate(sub_results)
    return measure


@cli.command()
@click.option(
    "--out-dir", default=None, type=click.Path(file_okay=False, dir_okay=True)
)
@click.option(
    "--load-dir", default=None, type=click.Path(file_okay=False, dir_okay=True)
)
@click.option("--ratio", default=0.8)
@click.option("--download-res", default="")
@click.option("--use-tuned-hparam", default="")
@click.option("--alpha", default=0.5)
@click.option("--beta", default=0.4)
@click.option("--neptune-on", is_flag=True)
def evaluate(
    out_dir,
    load_dir,
    ratio,
    download_res,
    use_tuned_hparam,
    alpha,
    beta,
    neptune_on,
):
    args = locals()
    tags = [f"alpha={alpha}", f"beta={beta}"]
    if neptune_on:
        proj_name = os.environ["NEPTUNE_PROJECT"]
        run = neptune.init(
            proj_name,
            name="evaluate_bootstrap",
            capture_hardware_metrics=False,
            tags=["bootstrap"],
        )
        run["params"] = args

    if load_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_dir = os.path.join(out_dir, timestamp)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        out_dir = load_dir

    model_infos = get_model_info("mmdet")

    eval_options = ["--eval-options"] + [
        f"otc_params=[(alpha, {alpha}), (beta, {beta}), (use_dummy, True)]"
    ]

    for model_cfg in MODEL_CFGS.keys():

        if not os.path.exists(
            os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py")
        ):
            download("mmdet", [model_cfg])

        model_info = model_infos.loc[model_cfg]
        checkpoint_name = os.path.basename(model_info.weight)

        hparam_options = ()
        if len(use_tuned_hparam):
            hparam_options = load_hparam_neptune(model_cfg, use_tuned_hparam)

        out_pkl = f"{os.path.join(out_dir, MODEL_CFGS[model_cfg]+'.pkl')}"

        if len(download_res):
            results = load_results(download_res, MODEL_CFGS[model_cfg])
            mmcv.dump(results, out_pkl)

        if not os.path.exists(out_pkl):
            _ = test(
                package="mmdet",
                config=os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"),
                checkpoint=os.path.join(DEFAULT_CACHE_DIR, checkpoint_name),
                gpus=2,
                launcher="pytorch",
                other_args=(
                    "--out",
                    out_pkl,
                    "--work-dir",
                    f"{out_dir}",
                    "--cfg-options",
                    f"data.test.type=CocoOtcDataset",
                    "custom_imports.imports=[src.extensions.dataset.coco_custom]",
                    "custom_imports.allow_failed_imports=False",
                    *hparam_options,
                    *eval_options,
                ),
            )

        n_trials = 100
        print(f"run {n_trials} trials")
        measures = []

        progress = tqdm(total=n_trials)

        with ProcessPoolExecutor(10) as pool:
            futures = []

            for _ in range(n_trials):
                future = pool.submit(
                    eval_on_subset, out_pkl, model_cfg, ratio=ratio
                )
                futures.append(future)

            for future in as_completed(futures):
                progress.update(1)
                measures.append(future.result())

        out_file = os.path.join(
            out_dir, MODEL_CFGS[model_cfg] + f".{ratio}.measures.json"
        )
        json.dump(measures, open(out_file, "w"))
        if neptune_on:
            run[f"measures/{MODEL_CFGS[model_cfg]}"].upload(out_file)

    if neptune_on:
        generate_reports(out_dir, ratio, nptn_run=run)
        run.stop()
    else:
        generate_reports(out_dir, ratio)


if __name__ == "__main__":
    cli()
