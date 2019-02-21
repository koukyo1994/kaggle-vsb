import sys
import time
import logging
import datetime as dt

from pathlib import Path
from contextlib import contextmanager

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


def parse_dict(d):
    for k in d.keys():
        if isinstance(d[k], list):
            values = [parse_dict(v) for v in d[k]]
            d[k] = values
        elif d[k] == "None" or d[k] == "True" or d[k] == "False":
            d[k] = eval(d[k])
    return d


@contextmanager
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
    path.mkdir(exist_ok=True, parents=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(
        path / (dt.datetime.now().strftime("%Y-%m-%d-%H-%M-%S") + ".log"))
    sh = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        "%(asctime)s %(name)s %(levelname)s %(message)s")

    fh.setFormatter(formatter)
    sh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger
