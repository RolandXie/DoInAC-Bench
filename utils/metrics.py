import numpy as np


def backward_transfer(results):
    n_tasks = len(results)
    li = []
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)


def forward_transfer(results, random_results):
    n_tasks = len(results)
    li = []
    for i in range(1, n_tasks):
        li.append(results[i - 1][i] - random_results[i])

    return np.mean(li)


def forgetting(results):
    n_tasks = len(results)
    li = []
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)


def forward_recovery(results, initial_results):
    """
    Normalized forward recovery / FOR.

    FOR_i = (max_t r_{t,i} - r_{T,i}) / (max_t r_{t,i} - r_{0,i})
    where r_{0,i} is the initialized model accuracy on task i and r_{T,i}
    is the final-stage accuracy on task i.
    """
    if len(results) == 0:
        return np.nan

    stacked = np.asarray([initial_results] + list(results), dtype=float)
    initial = np.asarray(initial_results, dtype=float)
    final = np.asarray(results[-1], dtype=float)
    peak = np.max(stacked, axis=0)

    denom = peak - initial
    numer = peak - final
    with np.errstate(divide='ignore', invalid='ignore'):
        per_task = np.where(denom != 0, numer / denom, np.nan)
    return np.nanmean(per_task)

def average_i(results, i):
    assert(i < len(results[0]))
    return np.mean(results[i][:i+1])

def average_iplus1(results, i):
    assert(i < len(results[0]))
    return np.mean(results[i][:i+2])
