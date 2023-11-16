import numpy as np
from numpy.core import numerictypes as nt
from numba import njit


def meandian(x, alpha=1.5, max_iter=100, tol=1e-10):
    # make sure the input is an array
    x = np.asarray(x)
    # if the exponent is a single value, return a singlevalue
    if not isinstance(alpha, (tuple, list, np.ndarray)):
        return meandian(x, np.array([alpha]))[0]

    alpha = np.asarray(alpha).astype(float)

    # make sure the input is not an int or bool
    if issubclass(x.dtype.type, (nt.integer, nt.bool_)):
        x = x.astype("f8")

    meandian_values = np.zeros(alpha.shape)
    if np.sum(alpha < 1):
        meandian_values[alpha < 1] = brute_meandian2(x, alpha[alpha < 1])

    if np.sum(alpha >= 1):
        meandian_values[alpha >= 1] = interval_meandian(
            x, alpha[alpha >= 1], max_iter=max_iter, tol=tol
        )

    return meandian_values


def get_meandian(x, alpha=1.5, max_iter=100, tol=1e-6, axis=None):
    single_alpha = False
    if isinstance(alpha, (int, float)):
        single_alpha = True
        alpha = [alpha]

    if len(x.shape) >= 2 and axis is None:
        x = x.ravel()

    if len(x.shape) >= 2:
        y = np.transpose(x, [i for i in range(len(x.shape)) if i != axis] + [axis])
        shape = y.shape
        yy = np.reshape(y, (-1, y.shape[-1]))

        mus = np.array(
            [interval_meandian_jit(xx, np.asarray(alpha), max_iter, tol) for xx in yy]
        )
        mus = np.reshape(mus, list(shape[:-1]) + [-1])
        if single_alpha is True:
            return mus[..., 0]
        return mus
    else:
        mus = interval_meandian_jit(x, np.asarray(alpha), max_iter, tol)

        if single_alpha is True:
            return mus[0]
        return mus


@njit
def interval_meandian_jit(x, alpha=1.5, max_iter=100, tol=1e-6):
    results = []

    for a in alpha:
        if np.isinf(a):
            results.append((np.max(x) + np.min(x)) / 2)
        elif a < 1:

            def d(x, mu, n, axis=0):
                x = x[:, None]
                mu = mu[None, :]
                return np.sum((np.abs(x - mu)) ** n, axis=axis)

            v = d(x, x, a)
            i = np.argmin(v, axis=0)
            results.append(x[i])
        elif a == 1:
            results.append(np.median(x))
        else:

            def dp(mu):
                a_mu = x - mu
                return np.sum(-np.abs(a_mu) ** (a - 1) * np.sign(a_mu))

            interval_x = [np.min(x), np.max(x)]
            for i in range(max_iter):
                new_point = (interval_x[0] + interval_x[1]) / 2
                value = dp(new_point)
                if value > 0:
                    interval_x[1] = new_point
                else:
                    interval_x[0] = new_point
                if np.abs(interval_x[0] - interval_x[1]) < tol:
                    break
            results.append((interval_x[0] + interval_x[1]) / 2)

    return np.array(results)


def interval_meandian(x, alpha=1.5, max_iter=100, tol=1e-10):
    if not isinstance(alpha, (tuple, list, np.ndarray)):
        alpha = np.array([alpha])

    def dp(mu):
        a_mu = x[:, None] - mu[None, :]
        return np.sum(-np.abs(a_mu) ** (alpha - 1) * np.sign(a_mu), axis=0)

    interval_x = np.array([[np.min(x), np.max(x)]] * len(alpha)).T
    range_n = np.arange(len(alpha), dtype=int)
    for i in range(max_iter):
        new_point = np.mean(interval_x, axis=0)
        value = dp(new_point)
        index = ((value > 0).astype(np.uint8), range_n)
        interval_x[index] = new_point
        if np.all(np.abs(interval_x[0] - interval_x[1]) < tol):
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


from mpmath import *

mp.dps = 50

# print(mpf(2) ** mpf("0.5"))


def optimize(x, alphas):
    x = np.sort(x)
    meandians = get_meandian(x, alphas)
    y = np.random.normal(size=len(x))
    losses = np.abs(get_meandian(y, alphas) - meandians)
    loss = np.sum(losses)
    list_meandians = []
    list_y = []
    list_lr = []
    lr = 0.01
    count = 0
    for i in range(1000_000):
        dy = np.random.normal(size=len(x)) * lr
        y2 = y + dy

        meandians2 = get_meandian(y2, alphas)
        losses2 = np.abs(meandians2 - meandians)
        loss2 = np.sum(losses2)
        if np.sum(loss2) < loss * 1.01 and np.sum(losses2 < losses) > 3:
            y2 = np.sort(y2)
            y = y2
            loss = np.sum(loss2)
            list_meandians.append(meandians2)
            list_y.append(y)
            list_lr.append(lr)
        else:
            count += 1
            if count > 100:
                lr *= 0.999
                count = 0

    list_y = np.array(list_y)
    list_meandians = np.array(list_meandians)

    def plot_both(target, values):
        for i in range(len(target)):
            (l,) = plt.plot(values[:, i])
            plt.axhline(target[i], color=l.get_color(), lw=0.8)

    plt.subplot(121)
    plot_both(meandians, list_meandians)
    plt.plot(list_lr)
    plt.subplot(122)
    plot_both(x, list_y)

    plt.show()


import matplotlib.pyplot as plt

np.random.seed(123)
x = np.random.normal(size=7)
alphas = np.array([0.5, 1, 1.3, 1.5, 1.8, 2, 2.5, 3, 5, np.inf])
# alphas = np.array([0.5])
optimize(x, alphas)

med0 = get_meandian(x, alphas)
off = []
dxs = []
for i in range(10000):
    dx = np.random.normal(size=len(x)) * 1e-2
    med = get_meandian(x + dx, alphas)
    off.append(np.sum(np.abs(med - med0)))
    dxs.append(np.sum(np.abs(dx)))
print(np.mean(off))


# plt.hist(off, 20)
plt.plot(off, dxs, "o", alpha=0.1)
plt.xlabel("meandian error")
plt.ylabel("data error")
plt.show()


def interval_meandian_mpmath(x, alpha=1.5, max_iter=100, tol=1e-6):
    results = []

    for a in alpha:
        if np.isinf(a):
            results.append((np.max(x) + np.min(x)) / 2)
        elif a < 1:

            def d(x, mu, n, axis=0):
                x = x[:, None]
                mu = mu[None, :]
                return np.sum((np.abs(x - mu)) ** n, axis=axis)

            v = d(x, x, a)
            i = np.argmin(v, axis=0)
            results.append(x[i])
        elif i == 1:
            results.append(np.median(x))
        else:

            def dp(mu):
                a_mu = x - mu
                return np.sum(-np.abs(a_mu) ** (a - 1) * np.sign(a_mu))

            interval_x = [np.min(x), np.max(x)]
            for i in range(max_iter):
                new_point = (interval_x[0] + interval_x[1]) / 2
                value = dp(new_point)
                if value > 0:
                    interval_x[1] = new_point
                else:
                    interval_x[0] = new_point
                if np.abs(interval_x[0] - interval_x[1]) < tol:
                    break
            results.append((interval_x[0] + interval_x[1]) / 2)

    return np.array(results)
