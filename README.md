# Optimal Correction Cost for Object Detection Evaluation

This repository is the official implementation of [Optimal Correction Cost for Object Detection Evaluation](https://arxiv.org/abs/2203.14438). 

[paper](https://openaccess.thecvf.com/content/CVPR2022/html/Otani_Optimal_Correction_Cost_for_Object_Detection_Evaluation_CVPR_2022_paper.html) | [video](https://www.dropbox.com/s/0ckbxd2odzv2znf/oc-cost_1.mp4?dl=0) | [poster](https://www.dropbox.com/s/w9duuk9q6wsi1do/Poster4.2-233b.pdf?dl=0) |
[blog (Japanese)](https://cyberagent.ai/blog/research/computervision/16366/) | [日経ロボティクス(Japanese)](https://xtech.nikkei.com/atcl/nxt/mag/rob/18/012600001/00101/) 

## Requirements

To install requirements:

```setup
poetry install
mim install mmcv-full
mim install mmdet
```

This code is tested on mmcv-full==1.3.10 and mmdet==2.15.0.

## Quick demo

You can try OC-cost on a notebook `notebooks/interactive_oc_demo.ipynb`.

## Data

If you want to test OC-cost on COCO, download [coco2017](https://cocodataset.org/#download) in `data` folder

```
data
├── annotations
└── val2017
```

## Evaluation

To evaluate detectors on COCO, run:

```eval
python src/tools/run_evaluation.py evaluate outputs/run_evaluation/ N_GPUs -s --use-tuned-hparam alpha=0.5,beta=0.6
```

The scirpt will download detectors from MMDetection and compute mAP and OC-cost on COCO validation 2017.

## Results
OC-cost and mAP of the detectors on MMDetection on COCO validation 2017 are as follows :

### OC-cost and mAP on COCO validation 2017

| Model name         |        mAP ↑     |    OC-cost ↓    |
| ------------------ |---------------- | -------------- |
| Faseter-RCNN [[config](https://github.com/open-mmlab/mmdetection/blob/master/configs/fast_rcnn/fast_rcnn_r50_fpn_2x_coco.py)]   |    0.38         |      0.45       |
|RetinaNet [[config](https://github.com/open-mmlab/mmdetection/blob/master/configs/retinanet/retinanet_r50_fpn_2x_coco.py)]   |    0.32         |      0.28       |
|DETR [[config](https://github.com/open-mmlab/mmdetection/blob/master/configs/detr/detr_r50_8x2_150e_coco.py)]   |     0.40     |    0.57     |
|YOLOF [[config](https://github.com/open-mmlab/mmdetection/blob/master/configs/yolof/yolof_r50_c5_8x8_1x_coco.py)]   |     0.32      |   0.30    |
|VFNet [[config](https://github.com/open-mmlab/mmdetection/blob/master/configs/vfnet/vfnet_r50_fpn_mstrain_2x_coco.py)]   |   0.37     |    0.26     |

NMS parameters are tuned on OC-cost.

## Citation

If this work helps your research, please cite:

```
@InProceedings{Otani_2022_CVPR,
    author    = {Otani, Mayu and Togashi, Riku and Nakashima, Yuta and Rahtu, Esa and Heikkil\"a, Janne and Satoh, Shin'ichi},
    title     = {Optimal Correction Cost for Object Detection Evaluation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2022},
    pages     = {21107-21115}
}
```