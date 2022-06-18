import tempfile

try:
    import neptune.new as neptune
except ImportError:
    raise ImportWarning("neptune client is not installed")

import os
from data.conf.model_cfg import HPARAM_RUNS, HPARAMS
import mmcv
from typing import Tuple


def load_hparam_cfg(model_cfg: str, key: str) -> Tuple[str]:
    """Load hyperparameter settings from config file

    Args:
        model_cfg (str): Model config name.
        key (str): Hyperparameter config name.

    Returns:
        Tuple[str]: Config strings
    """
    if model_cfg not in HPARAMS[key]:
        return []
    hparams = HPARAMS[key][model_cfg]

    score_thr = hparams["score_thr"]
    iou_threshold = hparams["iou_threshold"]
    hparam_options = (
        f"model.test_cfg.score_thr={score_thr}",
        f"model.test_cfg.nms.iou_threshold={iou_threshold}",
    )
    return hparam_options


def load_hparam_neptune(model_cfg: str, key: str) -> Tuple[str]:
    """Download hyperparameter settings from neptune

    Args:
        model_cfg (str): Model config name.
        key (str): Hyperparameter config name.

    Returns:
        Tuple[str]: Config strings
    """
    if model_cfg not in HPARAM_RUNS[key]:
        return []
    run_id = HPARAM_RUNS[key][model_cfg]
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
    with tempfile.TemporaryDirectory() as tmpdirname:
        run[f"results/{model_name}"].download(tmpdirname)
        results = mmcv.load(os.path.join(tmpdirname, f"{model_name}.pkl"))
    run.stop()
    return results
