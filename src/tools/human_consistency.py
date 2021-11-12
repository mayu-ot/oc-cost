import csv
import os
import numpy as np

from mim.utils import DEFAULT_CACHE_DIR
import os
import mmcv
from mmcv.utils.config import Config
import numpy as np
import pandas as pd
from src.extensions.metrics.ot_cost import get_ot_cost, get_cmap
from src.utils import krippendorff
from src.utils import map_single

from tqdm import tqdm
import seaborn as sns
from mmdet.datasets import build_dataset, get_loading_pipeline
from src.extensions.dataset.coco_custom import CocoOtcDataset  # workaround


# %%
def load_human_ann():
    ab2idx = {"A": 0, "B": 1}
    data = []
    for k in ["user_1", "user_2", "user_3"]:
        base_dir = f"data/raw/user_study/{k}/"
        user_judge = []
        for j in np.arange(1, 1100, 100):
            fn = f"{j}-{j+99}.csv"
            user_judge += [
                ab2idx.get(row[-1], np.nan)
                for row in csv.reader(open(f"{base_dir}/{fn}"))
            ]
        data.append(user_judge)
    return data


# %%
def preprocess_data(data):
    alpha = krippendorff.alpha(
        reliability_data=data, level_of_measurement="nominal"
    )
    print(f"all: {alpha:.3}")

    skip_user = 0
    alpha_max = 0
    for skip in range(len(data)):
        one_out_data = [v for i, v in enumerate(data) if i != skip]
        alpha = krippendorff.alpha(
            reliability_data=one_out_data, level_of_measurement="nominal"
        )
        print(f"skip-{skip}: {alpha:.3}")
        if alpha > alpha_max:
            alpha_max = alpha
            skip_user = skip
    print("skip user", skip_user)
    data = [v for i, v in enumerate(data) if i != skip_user]
    alpha = krippendorff.alpha(
        reliability_data=data, level_of_measurement="nominal"
    )
    print("validation", alpha)

    input_data = [
        row[:2] for row in csv.reader(open("data/raw/user_study/inputs.csv"))
    ]
    with open("data/raw/user_study/image_names.txt") as f:
        img_names = [line.split(".")[0] for line in f]

    selected_dets = []
    for v in data:
        selected_det = []
        for x, choices in zip(v, input_data):
            if not np.isnan(x):
                selected_det.append(choices[x])
            else:
                selected_det.append(np.nan)
        selected_dets.append(selected_det)
    selected_dets = np.asarray(selected_dets)

    N = len(selected_dets[0])
    is_agree = [
        i for i in range(N) if selected_dets[0][i] == selected_dets[1][i]
    ]
    selected_dets = selected_dets[0, is_agree]
    img_names = [img_names[i] for i in is_agree]
    print(
        f"yolof {(selected_dets=='yolof').sum()} vs retinanet {(selected_dets=='retinanet').sum()}"
    )

    anno_data = pd.DataFrame({"img": img_names, "selected_det": selected_dets})
    return anno_data


# %%
def prepare_data():
    retina_res = mmcv.load(
        "outputs/otc_search_param_candidates/20211027_003051/RetinaNet.pkl"
    )
    yolof_res = mmcv.load(
        "outputs/otc_search_param_candidates/20211027_003051/YOLOF.pkl"
    )
    model_cfg = "retinanet_r50_fpn_2x_coco"

    cfg = Config.fromfile(os.path.join(DEFAULT_CACHE_DIR, model_cfg + ".py"))
    cfg.data.test.type = "CocoOtcDataset"
    cfg.data.test.test_mode = True
    cfg.data.test.pop("samples_per_gpu", 0)
    cfg.data.test.pipeline = get_loading_pipeline(cfg.data.train.pipeline)
    dataset = build_dataset(cfg.data.test)

    selected_indexes = [
        dataset.img_ids.index(int(x)) for x in anno_data["img"].values
    ]
    retina_res = [retina_res[i] for i in selected_indexes]
    yolof_res = [yolof_res[i] for i in selected_indexes]
    return retina_res, yolof_res, dataset, selected_indexes


if __name__ == "__main__":
    data = load_human_ann()
    anno_data = preprocess_data(data)
    retina_res, yolof_res, dataset, selected_indexes = prepare_data()

    eval_fn = map_single.bbox_map_eval
    n = len(selected_indexes)
    retina_map = [
        eval_fn(res, dataset.prepare_train_img(i)["ann_info"])
        for i, res in tqdm(zip(selected_indexes, retina_res), total=n)
    ]
    yolof_map = [
        eval_fn(res, dataset.prepare_train_img(i)["ann_info"])
        for i, res in tqdm(zip(selected_indexes, yolof_res), total=n)
    ]
    map_judge = [
        ["yolof", "retinanet"][x > y] for x, y in zip(retina_map, yolof_map)
    ]
    acc = np.mean(np.asarray(map_judge) == anno_data["selected_det"].values)
    df = pd.DataFrame(
        {
            "human_judgment": anno_data["selected_det"].values,
            "map_judgment": map_judge,
        }
    )
    df.to_csv("outputs/human_consistency/map_vs_human.csv")
    print("mAP acc", acc)
