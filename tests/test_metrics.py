import numpy as np
from src.extensions.metrics.ot_cost import bbox_gious

a_bboxes = np.array([
    [0, 0, 10, 10],
    [10, 10, 20, 20],
    [32, 32, 38, 42],
])

b_bboxes = np.array([[0, 0, 10, 20], [0, 10, 10, 19], [10, 10, 20, 20]])

10 * 10 / 10*20
def test_gious():
    0.5, = bbox_gious(a_bboxes, b_bboxes)