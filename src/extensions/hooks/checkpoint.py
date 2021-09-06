from mmcv.runner.hooks.checkpoint import CheckpointHook
from dataclasses import dataclass
from mmcv.runner.hooks.hook import HOOKS
from mmcv.runner.dist_utils import master_only, allreduce_params
import subprocess
import os.path as osp


@HOOKS.register_module()
@dataclass
class AIPlatformCheckpointHook(CheckpointHook):
    def __init__(
        self,
        interval=-1,
        by_epoch=True,
        save_optimizer=True,
        out_dir=None,
        max_keep_ckpts=-1,
        save_last=True,
        sync_buffer=False,
        job_dir=None,
        **kwargs,
    ):
        self.interval = interval
        self.by_epoch = by_epoch
        self.save_optimizer = save_optimizer
        self.out_dir = out_dir
        self.max_keep_ckpts = max_keep_ckpts
        self.save_last = save_last
        self.args = kwargs
        self.sync_buffer = sync_buffer
        self.job_dir = job_dir

    @master_only
    def _upload_checkpoint(self, runner):
        if self.by_epoch:
            cur_ckpt_filename = self.args.get(
                "filename_tmpl", "epoch_{}.pth"
            ).format(runner.epoch + 1)
        else:
            cur_ckpt_filename = self.args.get(
                "filename_tmpl", "iter_{}.pth"
            ).format(runner.iter + 1)

        cur_ckpt_path = osp.join(self.out_dir, cur_ckpt_filename)
        subprocess.run(
            [
                "gsutil",
                "-q",
                "cp",
                "-r",
                cur_ckpt_path,
                self.job_dir + "/",
            ]
        )

    def after_train_iter(self, runner):
        if self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` iterations
        # 2. reach the last iteration of training
        if self.every_n_iters(runner, self.interval) or (
            self.save_last and self.is_last_iter(runner)
        ):
            runner.logger.info(
                f"Saving checkpoint at {runner.iter + 1} iterations"
            )
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)
            self._upload_checkpoint(runner)

    def after_train_epoch(self, runner):
        if not self.by_epoch:
            return

        # save checkpoint for following cases:
        # 1. every ``self.interval`` epochs
        # 2. reach the last epoch of training
        if self.every_n_epochs(runner, self.interval) or (
            self.save_last and self.is_last_epoch(runner)
        ):
            runner.logger.info(
                f"Saving checkpoint at {runner.epoch + 1} epochs"
            )
            if self.sync_buffer:
                allreduce_params(runner.model.buffers())
            self._save_checkpoint(runner)
            self._upload_checkpoint(runner)