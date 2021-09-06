from mmcv.runner.hooks.hook import HOOKS, Hook
from dataclasses import dataclass
from mmcv.runner.dist_utils import master_only
import hypertune


@HOOKS.register_module()
@dataclass
class HPTuneHook(Hook):
    metric_name: str

    def get_metric(self, runner):
        for var, val in runner.log_buffer.output.items():
            if var == self.metric_name:
                return val

    @master_only
    def _report_metric(self, runner):
        metric = self.get_metric(runner)
        epoch = runner.epoch
        hpt = hypertune.HyperTune()
        hpt.report_hyperparameter_tuning_metric(
            hyperparameter_metric_tag=self.metric_name,
            metric_value=metric,
            global_step=epoch,
        )

    def after_train_epoch(self, runner):
        self._report_metric(runner)