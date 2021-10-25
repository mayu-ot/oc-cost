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

from traitlets.traitlets import default
from src.extensions.dataset.coco_custom import CocoOtcDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob
import pandas as pd
from tabulate import tabulate

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
def generate_reports(
    out_dir, ratio, measures=["bbox_mAP", "bbox_mAP_50", "bbox_mAP_75", "mOTC"]
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


def eval_on_subset(out_pkl, model_cfg, ratio=0.8):
    results = mmcv.load(out_pkl)
    cfg = Config.fromfile(os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"))
    # write subset json file to tmp file
    dataset = json.load(
        open("data/coco/annotations/instances_val2017.json", "r")
    )
    n_dataset = len(dataset["images"])
    n_sub = int(n_dataset * ratio)

    rng = np.random.default_rng()
    sub_idx = rng.permutation(n_dataset)[:n_sub]

    sub_results = [results[i] for i in sub_idx]

    dataset["images"] = [dataset["images"][i] for i in sub_idx]

    tmp_dir = tempfile.TemporaryDirectory()
    tmp_file = os.path.join(tmp_dir.name, "sub_dataset.json")
    json.dump(dataset, open(tmp_file, "w"))

    coco = CocoOtcDataset(
        tmp_file,
        cfg.data.test.pipeline,
        test_mode=True,
    )
    measure = coco.evaluate(sub_results)
    return measure


@cli.command()
@click.option(
    "--out-dir", default=None, type=click.Path(file_okay=False, dir_okay=True)
)
@click.option(
    "--load-dir", default=None, type=click.Path(file_okay=False, dir_okay=True)
)
@click.option("--ratio", default=0.8)
def evaluate(out_dir, load_dir, ratio):
    if load_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_dir = os.path.join(out_dir, timestamp)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        out_dir = load_dir

    model_infos = get_model_info("mmdet")

    for model_cfg in MODEL_CFGS.keys():

        if not os.path.exists(
            os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py")
        ):
            download("mmdet", [model_cfg])

        model_info = model_infos.loc[model_cfg]
        checkpoint_name = os.path.basename(model_info.weight)

        # test hyperparameters
        # hparams = json.load(
        #     open(
        #         f"data/processed/tune_hparams_otc/{MODEL_CFGS[model_cfg]}_tune_res.json"
        #     )
        # )
        # score_thr = hparams["best_params"]["score_thr"]
        # iou_threshold = hparams["best_params"]["iou_threshold"]

        out_pkl = f"{os.path.join(out_dir, MODEL_CFGS[model_cfg]+'.pkl')}"
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
                    # f"model.test_cfg.score_thr={score_thr}",
                    # f"model.test_cfg.nms.iou_threshold={iou_threshold}",
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
            out_dir, MODEL_CFGS[model_cfg] + f"{ratio}.measures.json"
        )
        json.dump(measures, open(out_file, "w"))


if __name__ == "__main__":
    cli()
