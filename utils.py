import numpy as np


def pointwise_euclidean_distance(lm1, lm2):
    """
    Average pointwise euclidean distance between two landmarks
    :param lm1: 2D (num points, coords) or 3D (frame, num points, coords)
    :param lm2: 2D (num points, coords) or 3D (frame, num points, coords)
    :return:
    """
    assert lm1.shape == lm2.shape, "Not implemented for lm of different shapes"
    dist = np.mean(np.sqrt(np.sum(np.power(lm1 - lm2, 2), axis=-1)))
    return dist

