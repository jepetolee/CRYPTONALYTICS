from numba import jit


@jit
def cal_varience_percent(x, y):
    return 100 - 100 * (x / y)
