import neptune.new as neptune
import os
from data.conf.model_cfg import HPARAM_RUNS


def load_hparam_neptune(model_cfg):
    if model_cfg not in HPARAM_RUNS:
        return []
    run_id = HPARAM_RUNS[model_cfg]
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
