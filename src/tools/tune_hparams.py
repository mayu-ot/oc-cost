from mim import test, download, get_model_info
from mim.utils import DEFAULT_CACHE_DIR
import os
import json
import click
import time
import glob
import optuna
import neptune.new as neptune
import neptune.new.integrations.optuna as optuna_utils

MODEL_CFGS = {
    # "retinanet_r50_fpn_2x_coco": "RetinaNet",
    # "faster_rcnn_r50_fpn_2x_coco": "Faster-RCNN",
    # "yolof_r50_c5_8x8_1x_coco": "YOLOF",
    # "detr_r50_8x2_150e_coco": "DETR",
    "vfnet_r50_fpn_mstrain_2x_coco": "VFNet",
}


@click.group()
def cli():
    pass


@cli.command()
@click.argument("dataset")
@click.argument("out_dir", type=click.Path(file_okay=False, dir_okay=True))
def hptune(dataset, out_dir):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    out_dir = os.path.join(out_dir, timestamp)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    model_infos = get_model_info("mmdet")

    for model_cfg in MODEL_CFGS.keys():
        run = neptune.init(
            project=os.environ["NEPTUNE_PROJECT"],
            name="tune_hparams",
            tags=["optuna", "hptune", MODEL_CFGS[model_cfg]],
        )
        neptune_callback = optuna_utils.NeptuneCallback(run)

        if not os.path.exists(
            os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py")
        ):
            download("mmdet", [model_cfg])

        model_info = model_infos.loc[model_cfg]
        checkpoint_name = os.path.basename(model_info.weight)

        def objective(trial):
            score_thr = trial.suggest_float("score_thr", 0.01, 0.9, log=False)
            iou_threshold = trial.suggest_float(
                "iou_threshold", 0.1, 0.9, log=False
            )

            is_success, _ = test(
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
                    f"data.test.img_prefix=data/coco/train2017/",
                    "data.test.ann_file=data/coco/annotations/instances_train2017_subset.json",  # subset of train for hptube
                    "data.workers_per_gpu=1",
                    "custom_imports.imports=[src.extensions.dataset.coco_custom]",
                    "custom_imports.allow_failed_imports=False",
                    f"model.test_cfg.score_thr={score_thr}",
                    f"model.test_cfg.nms.iou_threshold={iou_threshold}",
                    "--eval-options",
                    "eval_map=False",
                ),
            )

            if is_success:
                files = glob.glob(os.path.join(out_dir, "*.json"))
                latest_file = max(files, key=os.path.getctime)
                res = json.load(open(latest_file))
                cost = res["metric"]["mOTC"]
            else:
                cost = None

            return cost

        study = optuna.create_study(
            study_name=f"{MODEL_CFGS[model_cfg]}",
            direction="minimize",
        )  # Create a new study.
        study.optimize(objective, n_trials=30, callbacks=[neptune_callback])

        run.stop()


if __name__ == "__main__":
    cli()
