"""
Run this file for DROPE experiments.
"""

import pandas as pd
from drorl.dro import run_dro
import os, pickle
from drorl.types import (
    OptimConfig,
    OptMethod,
    FitPiMethod,
    FitOutcomeFnMethod,
    AlphaUpdateMethod,
)
from dataclasses import replace


args = {
    "snips": OptimConfig(
        alpha_update_method=AlphaUpdateMethod.IPS, fit_pihat_method=None
    ),
    "xfitted_snips": OptimConfig(
        alpha_update_method=AlphaUpdateMethod.IPS,
        fit_pihat_method=FitPiMethod.XGBOOST,
        crossfit_num_folds=5,
    ),
    "sn_ldr": OptimConfig(
        alpha_update_method=AlphaUpdateMethod.LDR,
        fit_pihat_method=FitPiMethod.XGBOOST,
        fit_outcome_fn_method=FitOutcomeFnMethod.XGBOOST,
        crossfit_num_folds=5,
        run_ldml=True,
    ),
    "ground_truth_propensity_sn_ldr": OptimConfig(
        alpha_update_method=AlphaUpdateMethod.LDR,
        fit_pihat_method=None,
        fit_outcome_fn_method=FitOutcomeFnMethod.XGBOOST,
        crossfit_num_folds=5,
        run_ldml=True,
    ),
}

import itertools


dataset = "linear_5"
num_actions = 5


delta_list = [0.1, 0.2, 0.3]
results_path = "pkls/eval_results.pkl"
if os.path.exists(results_path):
    df = pd.read_pickle(results_path)
else:
    df = pd.DataFrame()
n_list = [1000, 2000, 3000, 4000, 5000, 8000, 10000, 15000, 20000]
seed_list = range(30)
for delta, n, seed in itertools.product(delta_list, n_list, seed_list):
    for algo, optim_config in args.items():
        df_dict = {
            "delta": delta,
            "n": n,
            "seed": seed,
            "algo": algo,
        }
        print(df_dict)
        if "delta" in df.columns:
            df_idx = df["delta"] == delta
            for k, v in df_dict.items():
                df_idx = df_idx & (df[k] == v)
            if df_idx.any():
                print("Already found!")
                continue

        data_path = f"data/{dataset}/{seed}/{n}.pkl"
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        optim_config = replace(
            optim_config,
            delta=delta,
            seed=seed,
            num_actions=num_actions,
        )
        if optim_config.alpha_update_method in [
            AlphaUpdateMethod.IPS,
            AlphaUpdateMethod.FULL_INFO,
        ]:
            for opt_method in [OptMethod.GRADIENT_ASCENT, OptMethod.NEWTONS]:
                res = run_dro(
                    data,
                    replace(optim_config, opt_method=opt_method),
                )
                if res is not None:
                    break
        else:
            res = run_dro(data, optim_config)

        if res is not None:
            df_dict.update(res)

        df = pd.concat([df, pd.DataFrame.from_records([df_dict])])

        df.to_pickle(results_path)
