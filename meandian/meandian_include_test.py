from meandian_include import meandian
import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st


@given(st.integers(1, 200), st.floats(1, 2))
def test_median(n, alpha):
    x = np.random.rand(n*2+1)
    # test the median
    np.testing.assert_allclose(np.median(x), meandian(x, 1))
    # test the mean
    np.testing.assert_allclose(np.mean(x), meandian(x, 2))
    # test if it throws an error
    meandian(x, alpha)

@given(st.integers(1, 200), st.floats(1, 2))
def test_median_int(n, alpha):
    x = np.random.randint(-10, 20, size=(n*2+1))
    # test the median
    np.testing.assert_allclose(np.median(x), meandian(x, 1))
    # test the mean
    np.testing.assert_allclose(np.mean(x), meandian(x, 2))
    # test if it throws an error
    meandian(x, alpha)
