from numba import jit, njit

@jit
def cal_varience_percent(x):
    return [100 - 100 * float(x[i + 1] / x[i]) for i in range(len(x) - 1)]
