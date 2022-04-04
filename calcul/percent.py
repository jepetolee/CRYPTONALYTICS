from numba import jit, njit

@jit
def cal_varience_percent(x):
    return [-100 + 100 * float(x[i + 1] / x[i]) for i in range(len(x) - 1)]

@jit
def varience_percent(a,b):
    return [-100 + 100 * float(b / a)]
