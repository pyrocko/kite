import numpy as num

from kite import util


def test_trim_matrix():
    arr = num.full((100, 100), num.nan)

    arr[-1, -1] = 1.0
    assert util.trimMatrix(arr).shape == (1, 1)

    arr[-2, -2] = 1.0
    assert util.trimMatrix(arr).shape == (2, 2)

    arr[num.diag_indices_from(arr)] = 1.0
    assert util.trimMatrix(arr).shape == arr.shape
