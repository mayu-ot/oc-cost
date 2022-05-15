import numpy as np
from src.extensions.metrics.ot_cost import bbox_gious

a_bboxes = np.array(
    [
        [0, 0, 10, 10],
        [10, 10, 20, 20],
        [32, 32, 38, 42],
    ]
)

b_bboxes = np.array([[0, 0, 10, 20], [0, 10, 10, 19], [10, 10, 20, 20]])

vals = np.asarray(
    [
        [0.5, 0, -0.5],
        [-0.25, -0.05, 1],
        [
            -(1596 - 260) / 1596,
            -(38 * 32 - 150) / (38 * 32),
            -(28 * 32 - 160) / (28 * 32),
        ],
    ]
)


def test_gious():
    gious = bbox_gious(a_bboxes, b_bboxes)
    assert gious.shape == (3, 3)
    assert np.isfinite(gious).all() == True
    assert np.isclose(gious, vals).all() == True
