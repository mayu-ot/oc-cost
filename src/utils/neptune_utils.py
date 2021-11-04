import neptune.new as neptune
import os
from data.conf.model_cfg import HPARAM_RUNS


def load_hparam_neptune(model_cfg, alpha=0.5, beta=0.4):
    if model_cfg not in HPARAM_RUNS[f"alpha={alpha},beta={beta}"]:
        return []
    run_id = HPARAM_RUNS[f"alpha={alpha},beta={beta}"][model_cfg]
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


def load_results(run_id, model_name):
    run = neptune.init(
        project=os.environ["NEPTUNE_PROJECT"], run=run_id, mode="read-only"
    )
    run[""]
