import sys
import time
import logging
import datetime as dt

from pathlib import Path

from sklearn.metrics import matthews_corrcoef


def threshold_search(y_true, y_proba):
    best_threshold = 0
    best_score = 0

    for threshold in [i * 0.01 for i in range(100)]:
        score = matthews_corrcoef(y_true, y_proba > threshold)
        if score > best_score:
            best_threshold = threshold
            best_score = score
    search_result = {"threshold": best_threshold, "mcc": best_score}
    return search_result


def min_max_transf(ts, min_data, max_data, range_needed=(-1, 1)):
    if min_data < 0:
        ts_std = (ts + abs(min_data)) / (max_data + abs(min_data))
    else:
        ts_std = (ts - min_data) / (max_data + abs(min_data))
    if range_needed[0] < 0:
        return ts_std * (
            range_needed[1] + abs(range_needed[0])) + range_needed[0]
    else:
        return ts_std * (range_needed[1] - range_needed[0]) + range_needed[0]


def timer(name, logger=None):
    t0 = time.time()
    yield
    msg = f"[{name}] done in {time.time()-t0:.0f} s"
    if logger:
        logger.info(msg)
    else:
        print(msg)


def get_logger(name="Main", tag="exp", log_dir="log/"):
    log_path = Path(log_dir)
    path = log_path / tag
    path.mkdir(exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log")
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
