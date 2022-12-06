import logging
import os
import pickle
import sys

import numpy as np


def configure_logging():
    logging.basicConfig(stream=sys.stdout, format="%(levelname)s:%(message)s", level=logging.INFO)


def ddict2dict(d):
    for k, v in d.items():
        if isinstance(v, dict):
            d[k] = ddict2dict(v)
    return dict(d)


def linspace(start, stop, num=50):
    """
    Linspace with additional points
    """
    grid = list(np.linspace(start, stop, num))
    step = (stop - start) / (num - 1)
    grid.extend([0.1 * step, 0.5 * step, stop - 0.1 * step, stop - 0.5 * step])
    return sorted(grid)


class PrintOnce:
    def __init__(self):
        self.printed = False

    def __call__(self, message):
        if not self.printed:
            print(message)
            self.printed = True


def load_or_calc_and_save(filename, ignore_if_exist=False):
    def my_decorator(func):
        def wrapped(n_graphs, n_params, n_jobs):
            if os.path.exists(filename):
                print(f"{func.__name__}: cache file {filename} found! Skip calculations")
                if not ignore_if_exist:
                    with open(filename, "rb") as f:
                        result = pickle.load(f)
                else:
                    result = None
            else:
                print(f"{func.__name__}: RECALC {filename}. n_graphs={n_graphs}, n_params={n_params}, n_jobs={n_jobs}")
                result = func(n_graphs=n_graphs, n_params=n_params, n_jobs=n_jobs)
                with open(filename, "wb") as f:
                    pickle.dump(result, f)
            return result

        return wrapped

    return my_decorator
