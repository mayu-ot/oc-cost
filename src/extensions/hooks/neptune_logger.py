# modification of mmcv.runner.hooks.logger.text.py by Open-MMLab.

from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.hooks.logger.base import LoggerHook
from mmcv.runner.dist_utils import master_only
import neptune
import os
from dataclasses import dataclass


@HOOKS.register_module()
@dataclass
class NeptuneLoggerHook(LoggerHook):
    """Logger hook for neptune.ai.

    In this logger hook, the information will be logged on neptune.ai.

    Args:
        by_epoch (bool): Whether EpochBasedRunner is used.
        interval (int): Logging interval (every k iterations).
        ignore_last (bool): Ignore the log of last iterations in each epoch
            if less than `interval`.
        reset_flag (bool): Whether to clear the output buffer after logging.
        interval_exp_name (int): Logging interval for experiment name. This
            feature is to help users conveniently get the experiment
            information from screen or log file. Default: 1000.
    """

    neptune_cfg: dict
    by_epoch: bool = True
    interval: int = 10
    ignore_last: bool = True
    reset_flag: bool = False
    interval_exp_name: int = 1000
    time_sec_tot = 0

    def _upload_cfg(self, runner):
        for fn in os.listdir(runner.work_dir):
            if fn.split(".")[-1] == "py":
                cfg_path = os.path.join(runner.work_dir, fn)
                break

        self.npt_exp.log_artifact(cfg_path)

    @master_only
    def before_run(self, runner):
        npt_cfg = self.neptune_cfg
        neptune.init(npt_cfg["project"])
        self.npt_exp = neptune.create_experiment(
            logger=runner.logger, **npt_cfg["exp_cfg"]
        )

        self._upload_cfg(runner)

    @master_only
    def log(self, runner):
        tags = self.get_loggable_tags(runner)
        for k, v in tags.items():
            self.npt_exp.log_metric(k, self.get_step(runner), v)

    def get_step(self, runner):  # copied from latest repo
        """Get the total training step/epoch."""
        return self.get_iter(runner)
