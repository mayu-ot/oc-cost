from dataclasses import dataclass
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
import numpy as np
from src.extensions.metrics.ot_cost import get_ot_cost, get_cmap
from copy import deepcopy
import pdb
import json
import time
import os.path as osp
from mmcv.utils import get_logger
import matplotlib.pyplot as plt
import seaborn as sns
import neptune.new as neptune
from neptune.new.types import File
from mmcv.runner.dist_utils import master_only

N_COCOCLASSES = 80


def count_items(items):
    ns = []
    for x in items:
        if x is None:
            n = 0
        else:
            n = sum(map(len, x))
        ns.append(n)
    return ns


def get_stats(ot_costs, gts, results):
    mean = np.mean(ot_costs)
    std = np.std(ot_costs)

    n_gts = count_items(gts)
    n_preds = count_items(results)

    cov_gts = np.cov(ot_costs, n_gts)[0, 1]
    cov_preds = np.cov(ot_costs, n_preds)[0, 1]

    return {
        "mean": mean,
        "std": std,
        "cov_n-gts": cov_gts,
        "cov_n-preds": cov_preds,
    }


def draw_stats(ot_costs, gts, results):
    n_gts = count_items(gts)
    n_preds = count_items(results)
    figures = {}

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    sns.kdeplot(x=n_gts, y=ot_costs, fill=True, cmap="rocket", ax=axes[0])
    # axes[0].scatter(n_gts, ot_costs)
    axes[0].set_title("otc vs # GTs")
    sns.kdeplot(x=n_preds, y=ot_costs, fill=True, cmap="rocket", ax=axes[1])
    # axes[1].scatter(n_preds, ot_costs)
    axes[1].set_title("otc vs # Preds")
    figures["otc_vs_num_bb"] = fig

    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True)
    axes[0].hist(n_gts, bins=10)
    axes[0].set_title("# Ground truth boudning boxes")
    axes[1].hist(n_preds, bins=10)
    axes[1].set_title("# Prediction boudning boxes")
    figures["dist_n_bb"] = fig

    fig = plt.figure()
    plt.hist(ot_costs, bins=10)
    plt.title("OTC Distribution")
    figures["dist_otc"] = fig

    fig_src = {"ot_costs": ot_costs, "n_gts": n_gts, "n_preds": n_preds}

    return figures, fig_src


def write2json(ot_costs, file_names):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    json_file = osp.join(f"tmp/otc_{timestamp}.json")
    data = [(f_name, c) for f_name, c in zip(file_names, ot_costs)]
    json.dump(data, open(json_file, "w"))


def eval_ot_costs(gts, results, cmap_func):
    return [get_ot_cost(x, y, cmap_func) for x, y in zip(gts, results)]


@DATASETS.register_module()
class CocoOtcDataset(CocoDataset):
    def __init__(
        self,
        ann_file,
        pipeline,
        classes=None,
        data_root=None,
        img_prefix="",
        seg_prefix=None,
        proposal_file=None,
        test_mode=False,
        filter_empty_gt=True,
        nptn_project_id="",
        nptn_run_id="",
        nptn_metadata_suffix="",
    ):

        super().__init__(
            ann_file,
            pipeline,
            classes=classes,
            data_root=data_root,
            img_prefix=img_prefix,
            seg_prefix=seg_prefix,
            proposal_file=proposal_file,
            test_mode=test_mode,
            filter_empty_gt=filter_empty_gt,
        )

        self.nptn_project_id = nptn_project_id
        self.nptn_run_id = nptn_run_id
        self.nptn_on = False

        if (nptn_project_id != "") and (nptn_run_id != ""):
            self.nptn_metadata_suffix = nptn_metadata_suffix
            self.nptn_on = True

    def evaluate(
        self,
        results,
        metric="bbox",
        logger=None,
        jsonfile_prefix=None,
        classwise=False,
        proposal_nums=(100, 300, 1000),
        iou_thrs=None,
        metric_items=None,
        eval_map=True,
        otc_params=[("alpha", 0.5), ("beta", 0.6)],
    ):
        """Evaluate predicted bboxes. Overide this method for your measure.

        Args:
            results ([type]): outputs of a detector
            metric (str, optional): [description]. Defaults to "bbox".
            logger ([type], optional): [description]. Defaults to None.
            jsonfile_prefix ([type], optional): [description]. Defaults to None.
            classwise (bool, optional): [description]. Defaults to False.
            proposal_nums (tuple, optional): [description]. Defaults to (100, 300, 1000).
            iou_thrs ([type], optional): [description]. Defaults to None.
            metric_items ([type], optional): [description]. Defaults to None.
            eval_map (bool): Whether to evaluating mAP
            otc_params (list): OC-cost parameters.
                                alpha (lambda in the paper): balancing localization and classification costs.
                                beta: cost of extra / missing detections.
                                Defaults to [("alpha", 0.5), ("beta", 0.6)]

        Returns:
            dict[str, float]: {metric_name: metric_value}
        """
        if eval_map:
            eval_results = super().evaluate(
                results,
                metric=metric,
                logger=logger,
                jsonfile_prefix=jsonfile_prefix,
                classwise=classwise,
                proposal_nums=proposal_nums,
                iou_thrs=iou_thrs,
                metric_items=metric_items,
            )
        else:
            eval_results = {}

        otc_params = {k: v for k, v in otc_params}
        mean_otc = self.eval_OTC(results, **otc_params)
        eval_results["mOTC"] = mean_otc

        if self.nptn_on:
            self.upload_eval_results(eval_results)

        return eval_results

    def get_gts(self):
        gts = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            gt = self._ann2detformat(ann_info)
            if gt is None:
                gt = [
                    np.asarray([]).reshape(0, 5)
                    for _ in range(len(self.CLASSES))
                ]
            gts.append(gt)
        return gts

    @master_only
    def upload_eval_results(self, eval_results):
        nptn_run = neptune.init(
            project=self.nptn_project_id,
            run=self.nptn_run_id,
            mode="sync",
            capture_hardware_metrics=False,
        )
        for k, v in eval_results.items():
            nptn_run[f"evaluation/summary/{k}/{self.nptn_metadata_suffix}"] = v
        nptn_run.stop()

    @master_only
    def upload_otc_results(self, ot_costs, gts, results):
        nptn_run = neptune.init(
            project=self.nptn_project_id,
            run=self.nptn_run_id,
            mode="sync",
            capture_hardware_metrics=False,
        )

        file_names = [x["file_name"] for x in self.data_infos]
        otc_per_img = json.dumps(list(zip(file_names, ot_costs)))
        nptn_run[f"evaluation/otc/per_img/{self.nptn_metadata_suffix}"].upload(
            File.from_content(otc_per_img, extension="json")
        )

        for k, v in get_stats(ot_costs, gts, results).items():
            nptn_run[
                f"evaluation/otc/stats/{k}/{self.nptn_metadata_suffix}"
            ] = v

        figs, fig_src = draw_stats(ot_costs, gts, results)
        for fig_name, fig in figs.items():
            nptn_run[
                f"evaluation/figs/{fig_name}/{self.nptn_metadata_suffix}"
            ].upload(File.as_image(fig))
            fig.savefig(f"tmp/{fig_name}.pdf", bbox_inches="tight")
            nptn_run[
                f"evaluation/figs/pdfs/{fig_name}/{self.nptn_metadata_suffix}"
            ].upload(f"tmp/{fig_name}.pdf")

        nptn_run.stop()

    def eval_OTC(
        self,
        results,
        alpha=0.8,
        beta=0.4,
        get_average=True,
    ):
        gts = self.get_gts()
        cmap_func = lambda x, y: get_cmap(
            x, y, alpha=alpha, beta=beta, mode="giou"
        )
        tic = time.time()
        ot_costs = eval_ot_costs(gts, results, cmap_func)
        toc = time.time()
        print("OTC DONE (t={:0.2f}s).".format(toc - tic))

        if self.nptn_on:
            self.upload_otc_results(ot_costs, gts, results)

        if get_average:
            mean_ot_costs = np.mean(ot_costs)
            return mean_ot_costs
        else:
            return ot_costs

    def evaluate_gt(
        self,
        bbox_noise_level=None,
        **kwargs,
    ):

        gts = self.get_gts()
        n = len(gts)
        for i in range(n):
            gt = gts[i]
            if gt is None:
                gts[i] = [np.asarray([]).reshape(0, 5) for _ in self.CLASSES]
                continue
            for bbox in gt:
                if len(bbox) == 0:
                    continue

                w = bbox[:, 2] - bbox[:, 0]
                h = bbox[:, 3] - bbox[:, 1]
                shift_x = (
                    w * bbox_noise_level * np.random.choice((-1, 1), w.shape)
                )
                shift_y = (
                    h * bbox_noise_level * np.random.choice((-1, 1), h.shape)
                )
                bbox[:, 0] += shift_x
                bbox[:, 2] += shift_x
                bbox[:, 1] += shift_y
                bbox[:, 3] += shift_y
        return self.evaluate(gts, **kwargs)

    def _ann2detformat(self, ann_info):
        """convert annotation info of CocoDataset into detection output format.

        Parameters
        ----------
        ann : list[dict]
            ground truth annotation. each item in the list correnponds to an instance.
            >>> ann_info[i].keys()
            dict_keys(['segmentation', 'area', 'iscrowd', 'image_id', 'bbox', 'category_id', 'id'])
        Returns
        -------
        bboxes : list[numpy]
            list of bounding boxes with confidence score.
            bboxes[i] contains bounding boxes of instances of class i.
        """
        if len(ann_info) == 0:
            return None

        bboxes = [[] for _ in range(len(self.cat2label))]

        for ann in ann_info:
            if ann.get("ignore", False) or ann["iscrowd"]:
                continue
            c_id = ann["category_id"]
            x1, y1, w, h = ann["bbox"]

            bboxes[self.cat2label[c_id]].append([x1, y1, x1 + w, y1 + h, 1.0])

        np_bboxes = []
        for x in bboxes:
            if len(x):
                np_bboxes.append(np.asarray(x, dtype=np.float32))
            else:
                np_bboxes.append(
                    np.asarray([], dtype=np.float32).reshape(0, 5)
                )
        return np_bboxes
