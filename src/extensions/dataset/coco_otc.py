from dataclasses import dataclass
from mmdet.datasets.coco import CocoDataset
from mmdet.datasets.builder import DATASETS
import numpy as np
from src.extensions.metrics.ot_cost import (
    get_ot_cost,
    get_distmap,
    get_distmap_bg,
)
from copy import deepcopy
import pdb
import json
import time
import os.path as osp

N_COCOCLASSES = 80
print("imported!")


def write2json(ot_costs, file_names):
    timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    json_file = osp.join(f"tmp/otc_{timestamp}.json")
    data = [(f_name, c) for f_name, c in zip(file_names, ot_costs)]
    json.dump(data, open(json_file, "w"))


def eval_ot_costs(gts, results):
    cmap_func = lambda x, y: get_distmap_bg(x, y, mode="giou")
    return [get_ot_cost(x, y, cmap_func) for x, y in zip(gts, results)]


@DATASETS.register_module()
class CocoOtcDataset(CocoDataset):
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
    ):
        """Evaluate predicted bboxes. Override this method for your measure.

        Args:
            results ([type]): [description]
            metric (str, optional): [description]. Defaults to "bbox".
            logger ([type], optional): [description]. Defaults to None.
            jsonfile_prefix ([type], optional): [description]. Defaults to None.
            classwise (bool, optional): [description]. Defaults to False.
            proposal_nums (tuple, optional): [description]. Defaults to (100, 300, 1000).
            iou_thrs ([type], optional): [description]. Defaults to None.
            metric_items ([type], optional): [description]. Defaults to None.

        Returns:
            dict[str, float]: {metric_name: metric_value}
        """

        mean_otc = self.eval_OTC(results, logger=logger)
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
        eval_results["mOTC"] = mean_otc
        return eval_results

    def get_gts(self):
        gts = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)

            gts.append(self._ann2detformat(ann_info))
        return gts

    def eval_OTC(self, results, logger=None):
        gts = self.get_gts()
        ot_costs = eval_ot_costs(gts, results)
        file_names = [x["img_metas"][0].data["ori_filename"] for x in self]
        write2json(ot_costs, file_names)
        mean_ot_costs = np.mean(ot_costs)
        return mean_ot_costs

    def evaluate_gt(
        self,
        bbox_noise_level=None,
        cls_noise_level=None,
        **kwargs,
    ):

        gts = self.get_gts()
        for gt in gts:
            for bbox in gt:
                if len(bbox) == 0:
                    continue

                w = bbox[:, 2] - bbox[:, 0]
                h = bbox[:, 3] - bbox[:, 1]
                shift_x = (
                    w * bbox_noise_level * np.random.choice((-1, 1), w.shape)
                )
                shift_y = (
                    w * bbox_noise_level * np.random.choice((-1, 1), h.shape)
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
