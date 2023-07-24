from meandian_include import meandian
import numpy as np
from hypothesis import given
from hypothesis import strategies as st


@given(st.integers(1), st.floats(1, 2))
def test_median(n, alpha):
    x = np.random.rand(n+1)
    # test the median
    np.testing.assert_allclose(np.median(x), meandian(x, 1))
    # test the mean
    np.testing.assert_allclose(np.mean(x), meandian(x, 2))

def test_median_int():
    x = [1, 3, 5, 8, 8]
    # test the median
    np.testing.assert_allclose(np.median(x), meandian(x, 1))
    # test the mean
    np.testing.assert_allclose(np.mean(x), meandian(x, 2))
    #assert np.mean(x) <= meandian(x, alpha) <= np.mean(x, 2)