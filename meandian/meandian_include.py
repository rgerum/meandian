import numpy as np
from numpy.core import numerictypes as nt

def meandian(x, alpha=1.5):
    # make sure the input is an array
    x = np.asarray(x)
    # if the exponent is a single value, return a singlevalue
    if not isinstance(alpha, (tuple, list, np.ndarray)):
        return meandian(x, np.array([alpha]))[0]

    alpha = np.asarray(alpha).astype(float)

    # make sure the input is not an int or bool
    if issubclass(x.dtype.type, (nt.integer, nt.bool_)):
        x = x.astype('f8')

    meandian_values = np.zeros(alpha.shape)
    if np.sum(alpha < 1):
        meandian_values[alpha < 1] = brute_meandian2(x, alpha[alpha < 1])

    if np.sum(alpha >= 1):
        meandian_values[alpha >= 1] = interval_meandian(x, alpha[alpha >= 1])

    return meandian_values


def interval_meandian(x, alpha=1.5):
    if not isinstance(alpha, (tuple, list, np.ndarray)):
        alpha = np.array([alpha])

    def dp(mu):
        a_mu = x[:, None] - mu[None, :]
        return np.sum(-np.abs(a_mu) ** (alpha - 1) * np.sign(a_mu), axis=0)

    interval_x = np.array([[np.min(x), np.max(x)]] * len(alpha)).T
    range_n = np.arange(len(alpha), dtype=int)
    for i in range(100):
        new_point = np.mean(interval_x, axis=0)
        value = dp(new_point)
        index = ((value > 0).astype(np.uint8), range_n)
        interval_x[index] = new_point
        if np.all(np.abs(interval_x[0] - interval_x[1]) < 1e-10):
            break
    return np.mean(interval_x, axis=0)


def brute_meandian2(x, alpha=1.5):
    if isinstance(alpha, (np.ndarray, list, tuple)):
        return np.array([brute_meandian2(x, n0) for n0 in alpha])

    def d(x, mu, n, axis=0):
        x = x[:, None]
        mu = mu[None, :]
        return np.sum((np.abs(x - mu)) ** n, axis=axis)

    v = d(x, x, alpha)
    i = np.argmin(v, axis=0)
    return x[i]

