from mim import test, download, get_model_info
from mim.utils import DEFAULT_CACHE_DIR
import os
import json
import mmcv
from mmcv.utils.config import Config
import numpy as np
import click
import time

from traitlets.traitlets import default
from src.extensions.dataset.coco_custom import CocoOtcDataset
from concurrent.futures import ProcessPoolExecutor, as_completed
from tabulate import tabulate
import neptune.new as neptune
from neptune.new.types import File
import json
from itertools import compress

MODEL_CFGS = {
    "retinanet_r50_fpn_2x_coco": ("RetinaNet", "EV-23"),
    # "faster_rcnn_r50_fpn_2x_coco": "Faster-RCNN",
    "yolof_r50_c5_8x8_1x_coco": ("YOLOF", "EV-31"),
    # "detr_r50_8x2_150e_coco": "DETR",
    # "vfnet_r50_fpn_mstrain_2x_coco": "VFNet",
}


@click.group()
def cli():
    pass


def get_search_steps(out_pkl_a, out_pkl_b, data_cfg):
    results_a = mmcv.load(out_pkl_a)
    results_b = mmcv.load(out_pkl_b)
    out_dir = os.path.dirname(out_pkl_a)

    coco = CocoOtcDataset(
        "data/coco/annotations/instances_val2017.json",
        data_cfg.test.pipeline,
        test_mode=True,
    )

    file_names = [x["img_metas"][0].data["ori_filename"] for x in coco]
    df_a = {"image": file_names}
    df_b = {"image": file_names}

    params = {"alpha": 0.5, "beta": 0.4}
    eps = 0.02

    for p in np.arange(0.1, 0.99, 0.2):
        for q in np.arange(0.1, 0.99, 0.2):
            params["alpha"] = p
            params["beta"] = q

            ot_costs_a = coco.eval_OTC(results_a, **params, get_average=False)
            ot_costs_b = coco.eval_OTC(results_b, **params, get_average=False)

            df_a[
                f"alpha={params['alpha']:.1}_beata={params['beta']:.1f}"
            ] = ot_costs_a
            df_b[
                f"alpha={params['alpha']:.1}_beata={params['beta']:.1f}"
            ] = ot_costs_b

        #     is_a_better = [
        #         c_a + eps < c_b for c_a, c_b in zip(ot_costs_a, ot_costs_b)
        #     ]
        #     dicisions.append(is_a_better)

        # n_disagree = {}
        # for step in [1, 2, 3]:
        #     n_disagree[f"{0.1*step}"] = []
        #     for i in range(len(dicisions) - step):
        #         is_disagree = [x != y for x, y in zip(dicisions[i], dicisions[i + step])]
        #         disagree_files = compress(file_names, is_disagree)
        #         n = sum(is_disagree)
        #         n_disagree[f"{0.1*step}"].append(n)

        # json.dump(n_disagree, open(f"{out_dir}/{key}_steps.json", "w"))


def load_hparam(run_id):
    run = neptune.init(
        project=os.environ["NEPTUNE_PROJECT"], run=run_id, mode="read-only"
    )
    hparams = run["best/params"].fetch()

    score_thr = hparams["best"]["params"]["score_thr"]
    iou_threshold = hparams["best"]["params"]["iou_threshold"]
    hparam_options = (
        f"model.test_cfg.score_thr={score_thr}",
        f"model.test_cfg.nms.iou_threshold={iou_threshold}",
    )
    run.stop()
    return hparam_options


@cli.command()
@click.option(
    "--out-dir", default=None, type=click.Path(file_okay=False, dir_okay=True)
)
@click.option(
    "--load-dir", default=None, type=click.Path(file_okay=False, dir_okay=True)
)
@click.option("--neptune-on", is_flag=True)
def run(out_dir, load_dir, neptune_on):
    nptn_cfg = []
    nptn_run_id = ""
    if neptune_on:
        proj_name = os.environ["NEPTUNE_PROJECT"]
        run = neptune.init(proj_name)
        nptn_run_id = run._short_id
        nptn_cfg = [
            f"data.test.nptn_project_id={proj_name}",
            f"data.test.nptn_run_id={nptn_run_id}",
            "",
        ]

    if load_dir is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        out_dir = os.path.join(out_dir, timestamp)

        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
    else:
        out_dir = load_dir

    model_infos = get_model_info("mmdet")
    out_pkls = []
    for model_cfg in MODEL_CFGS.keys():

        if not os.path.exists(
            os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py")
        ):
            download("mmdet", [model_cfg])

        model_info = model_infos.loc[model_cfg]
        checkpoint_name = os.path.basename(model_info.weight)

        model_name, run_id = MODEL_CFGS[model_cfg]
        hparam_options = load_hparam(run_id)
        out_pkl = f"{os.path.join(out_dir, model_name+'.pkl')}"
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
                    "data.test.ann_file=data/coco/annotations/instances_val2017.json",
                    *hparam_options,
                ),
            )

        out_pkls.append(out_pkl)

    cfg = Config.fromfile(os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"))
    get_search_steps(*out_pkls, cfg.data)


if __name__ == "__main__":
    cli()
