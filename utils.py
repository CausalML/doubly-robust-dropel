import json
import pandas as pd
import glob
from dataclasses import fields
import numpy as np
import random
import jax
import os


@jax.jit
def adv_idx(mat, idx_arr):
    # Equiv to mat[:, idx_arr], since adv. indexing not available
    return (jax.nn.one_hot(idx_arr, num_classes=mat.shape[1]) * mat).sum(1)


def get_results_df(results_dir: str, extract_fn):
    """
    Apply extract_fn to the json.
    extract_fn should output a dict of important features.
    """
    paths = glob.glob(f"{results_dir}/*.out")
    print(f"Found {len(paths)} results from {results_dir}")
    df_dict = []
    for path in paths:
        with open(path) as f:
            d = json.load(f)
            keep = extract_fn(d)
            df_dict.append(keep)
    df = pd.DataFrame(df_dict)
    return df


def print_memory_util(datacls):
    total_memory = 0
    for field in fields(datacls):
        val = getattr(datacls, field.name)
        if val is not None:
            mem_usage = val.nbytes / (2 ** 10)
            print(
                f"{field.name} is numpy array of {val.size} {val.dtype} elems, taking {mem_usage} MBs"
            )
            total_memory += mem_usage
    print(f"Total memory: {total_memory} MBs")


def set_global_seeds(seed):
    """
    Sets global seeds
    """
    random.seed(seed)
    np.random.seed(seed)
    rng_key = jax.random.PRNGKey(seed)
    return rng_key


def subarray_datacls(x, subarray_len, from_back: bool = False):
    """All fields need to be length n"""
    d = {}
    n = None
    for f in fields(x):
        f = f.name
        v = getattr(x, f)
        if v is None:
            continue

        if n is None:
            n = v.shape[0]
        assert v.shape[0] == n
        if not from_back:
            d[f] = v[:subarray_len]
        else:
            d[f] = v[len(v) - subarray_len :]
    return type(x)(**d)


def vectorized_multinomial(prob_matrix):
    s = prob_matrix.cumsum(axis=1)
    r = np.random.rand(prob_matrix.shape[0], 1)
    k = (s < r).sum(axis=1).astype(np.int32)
    return k


def assert_no_nans(arr):
    assert np.all(
        ~np.isnan(arr)
    ), f"Arr has NaN at {np.where(np.isnan(arr))}. Arr: {arr}"


class DataFrameAggregator:
    def __init__(self, path):
        self.path = path
        if os.path.exists(path):
            self.df = pd.read_pickle(path)
            print(f"Read from {path} a dataframe containing {len(self.df)} entries.")
        else:
            self.df = pd.DataFrame()

    def exists(self, partial_row):
        """some subset of column/vals"""
        if len(self.df) == 0:
            return False

        idx = None
        for k, v in partial_row.items():
            if idx is None:
                idx = self.df[k] == v
            else:
                idx = idx & (self.df[k] == v)
        return idx.any()

    def append(self, new_row):
        """new_row is a dict"""
        self.df = pd.concat([self.df, pd.DataFrame.from_records([new_row])])
        self.df.to_pickle(self.path)
