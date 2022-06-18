MODEL_CFGS = {
    "retinanet_r50_fpn_2x_coco": "RetinaNet",
    "faster_rcnn_r50_fpn_2x_coco": "Faster-RCNN",
    "yolof_r50_c5_8x8_1x_coco": "YOLOF",
    "detr_r50_8x2_150e_coco": "DETR",
    "vfnet_r50_fpn_mstrain_2x_coco": "VFNet",
}

HPARAM_RUNS = {
    "alpha=0.5,beta=0.6": {
        "retinanet_r50_fpn_2x_coco": "EV-110",
        "faster_rcnn_r50_fpn_2x_coco": "EV-111",
        "yolof_r50_c5_8x8_1x_coco": "EV-112",
        "vfnet_r50_fpn_mstrain_2x_coco": "EV-113",
    },
    "mAP": {
        "retinanet_r50_fpn_2x_coco": "EV-85",
        "faster_rcnn_r50_fpn_2x_coco": "EV-86",
        "yolof_r50_c5_8x8_1x_coco": "EV-89",
        "vfnet_r50_fpn_mstrain_2x_coco": "EV-90",
    },
}

HPARAMS = {
    "alpha=0.5,beta=0.6": {
        "retinanet_r50_fpn_2x_coco": {
            "score_thr": 0.435989,
            "iou_threshold": 0.412228,
        },  # "EV-110"
        "faster_rcnn_r50_fpn_2x_coco": {
            "score_thr": 0.500332,
            "iou_threshold": 0.415385,
        },  # "EV-111"
        "yolof_r50_c5_8x8_1x_coco": {
            "score_thr": 0.378166,
            "iou_threshold": 0.53194,
        },  # "EV-112",
        "vfnet_r50_fpn_mstrain_2x_coco": {
            "score_thr": 0.533642,
            "iou_threshold": 0.618896,
        },  # "EV-113",
    },
    "mAP": {
        "retinanet_r50_fpn_2x_coco": {
            "score_thr": 0.0839853,
            "iou_threshold": 0.405454,
        },  # "EV-85",
        "faster_rcnn_r50_fpn_2x_coco": {
            "score_thr": 0.697227,
            "iou_threshold": 0.500017,
        },  # "EV-86",
        "yolof_r50_c5_8x8_1x_coco": {
            "score_thr": 0.0822186,
            "iou_threshold": 0.569493,
        },  # "EV-89",
        "vfnet_r50_fpn_mstrain_2x_coco": {
            "score_thr": 0.100739,
            "iou_threshold": 0.714145,
        },  # "EV-90",
    },
}
