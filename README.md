# Dependencies

1. install mim (https://github.com/open-mmlab/mim)
2. install mmcv-full and mmdetection using mim command (https://github.com/open-mmlab/mim#command)

other dependencies:

- click
- seaborn
- (optional. for OT cost evaluation) POT (https://pythonot.github.io/index.html)

# Dataset

## COCO

download coco2017 in 'data' folder

```
data
├── annotations
├── test2017
├── train2017
└── val2017
```

### (Note) Data split for evaluation

COCO recommends to use test-dev (20K) for evaluation.

However, we cannot access to ground truthes on the split.

[LRP paper (ECCV'18 paper)](https://openaccess.thecvf.com/content_ECCV_2018/papers/Kemal_Oksuz_Localization_Recall_Precision_ECCV_2018_paper.pdf) uses coco2017 validation set.

We can follow this, but be careful about its results:

- many public models use validation set for hyperparameter tuning / model selection
- the validation size is only 5k. Rare categories have very limited instances. (e.g., 9 instances for "toaster" class)

## How to run evaluation using your costom measure

### 1. make custom dataset to evaluate your measure

See `src/extensions/dataset/coco_otc.py`.

Override `DTASET.evaluate()` method which is called from `run_evaluation.py`.

### 2. import the dataset class in `src/src/extensions/dataset/coco_custom.py`

For example,

```
from .coco_otc import CocoOtcDataset
```

Run

```
PYTHONPATH='./' python src/tools/run_evaluation.py evaluate YOUR_CUSTOM_DATASET OUT_DIR
```

For OT cost evaluation, the command would be

```
PYTHONPATH='./' python src/tools/run_evaluation.py evaluate CocoOtcDataset outputs/otc_eval_res/
```

Quick visualization of the results is done by

```
python src/tools/run_evaluation.py generate-reports OUT_DIR
```