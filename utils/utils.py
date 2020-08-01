import csv
import math
import time
import numpy as np

from .types_ import *


def rmse(output: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Root Mean Square Error function
    ===============================

    Arguments
    ---------
    output : np.ndarray
    target : np.ndarray

    Returns
    -------
    rmse : np.ndarray
    """
    output = np.float64(output)
    target = np.float64(target)

    # sum square error (sse)
    sse = np.sum(np.square(target - output))
    return np.sqrt(np.mean(sse) / output.shape[0])


def time_since(since) -> str:
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return "%dm %ds" % (m, s)


def to_csv(file_name: str, preds: np.ndarray, testset: np.ndarray, header: bool = True) -> None:
    """
    Arguments
    ---------
    file_name : str
        file_name to be saved
    preds : np.ndarray
        model's output
    testset : np.ndarray
        test dataset
    header : bool
        use header or not

    Returns
    -------
    None
    """
    with open(file_name, "w", newline="") as f:
        wr = csv.writer(f)
        if header:
            wr.writerow(["userid", "movieid", "predicted rating", "timestamp"])
        for pred, info in zip(preds, testset[:, [0, 1, 3]]):
            user_id, item_id, ts = info
            wr.writerow([int(user_id), int(item_id), pred, int(ts)])
    return None
