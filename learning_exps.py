"""
Run this file for DROPL experiments.
"""
from drorl.dro import run_dro
import pickle
from drorl.types import (
    OptimConfig,
    OptMethod,
    FitPiMethod,
    FitOutcomeFnMethod,
    AlphaUpdateMethod,
)
from dataclasses import replace
import numpy as np

from utils import adv_idx, set_global_seeds, DataFrameAggregator
import itertools


import logging

logging.basicConfig(
    format="%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s",
    datefmt="%Y-%m-%d:%H:%M:%S",
    level=logging.INFO,
)
log = logging.getLogger(__name__)


## NOTE: toggle this to run baseline vs. CDR^2OPL.
run_name = "dro_baseline"
# run_name = "our_method"


max_iters = 500
num_seeds = 30
num_random_initializations = 1
dataset = "linear_5"
num_actions = 5
optimizer = "adam"
learning_rate = 0.01


mlp_hidden = [32]
mlp_temp = 1.0
learn_optim_config = dict(
    learn_pi=True,
    learn_mlp_hidden=mlp_hidden,
    learn_mlp_temp=mlp_temp,
    warmstart_mlp=False,
    converge_criterion=1e-4,
    num_consecutive_stops_needed=5,
    max_iters=max_iters,
    learn_mlp_epochs=1,
    learn_mlp_batch_size=1024,
    learn_mlp_lr=learning_rate,
    learn_mlp_optimizer=optimizer,
    learn_mlp_num_perturbations=9,
)

delta_list = [0.1, 0.2, 0.3]
n_list = [5000, 10000, 15000, 20000]
seed_list = range(num_seeds)

if run_name == "dro_baseline":
    results_path = "pkls/xfit_snips.pkl"
    args = {
        "xfit_snips": OptimConfig(
            alpha_update_method=AlphaUpdateMethod.IPS,
            crossfit_num_folds=5,
            fit_pihat_method=FitPiMethod.XGBOOST,
            opt_method=OptMethod.GRADIENT_ASCENT,
        ),
    }
else:
    results_path = "pkls/cdr.pkl"
    args = {
        "cdr_gradient": OptimConfig(
            alpha_update_method=AlphaUpdateMethod.LDR,
            crossfit_num_folds=5,
            fit_pihat_method=FitPiMethod.XGBOOST,
            fit_outcome_fn_method=FitOutcomeFnMethod.RANDOM_FOREST_CONTINUUM,
            # can use gradient since outcome function is open box, i.e. we can compute gradient
            opt_method=OptMethod.GRADIENT_ASCENT,
        ),
    }


df_agg = DataFrameAggregator(results_path)


def get_eval_results_for_policy(data, seed):
    eval_delta_list = [0.1, 0.2, 0.3, 0.5]
    ## First evaluate regular reward
    full_info_reward = (data.probs_mat * data.reward_mat).sum(1).mean()
    ipw = adv_idx(data.probs_mat, data.a) / data.a_prob
    ipw = ipw / ipw.mean()
    snips_est_reward = (ipw * data.r).mean()
    output_dict = {
        "full_info_reward": full_info_reward,
        "snips_est_reward": snips_est_reward,
    }

    for delta in eval_delta_list:
        full_info_dro = run_dro(
            data=data,
            optim_config=OptimConfig(
                seed=seed,
                delta=delta,
                alpha_update_method=AlphaUpdateMethod.FULL_INFO,
                fit_pihat_method=None,
                opt_method=OptMethod.GRADIENT_ASCENT,
                converge_criterion=1e-6,
                num_consecutive_stops_needed=20,
            ),
        )

        snips_dro = run_dro(
            data=data,
            optim_config=OptimConfig(
                seed=seed,
                delta=delta,
                alpha_update_method=AlphaUpdateMethod.IPS,
                fit_pihat_method=None,
                opt_method=OptMethod.GRADIENT_ASCENT,
                converge_criterion=1e-6,
                num_consecutive_stops_needed=20,
            ),
        )
        output_dict.update(
            {
                f"full_info_{delta}_dro_reward": full_info_dro["phi_n"],
                f"snips_{delta}_dro_reward": snips_dro["phi_n"],
                f"full_info_{delta}_dro_alpha": full_info_dro["alpha"],
                f"snips_{delta}_dro_alpha": snips_dro["alpha"],
            }
        )
    return output_dict


for n, delta, seed in itertools.product(n_list, delta_list, seed_list):
    test_data_path = f"data/{dataset}/{seed}/test.pkl"
    with open(test_data_path, "rb") as f:
        test_data = pickle.load(f)

    data_path = f"data/{dataset}/{seed}/{n}.pkl"
    with open(data_path, "rb") as f:
        data = pickle.load(f)

    for algo, optim_config in args.items():
        df_dict = {
            "delta": delta,
            "n": n,
            "seed": seed,
            "algo": algo,
        }
        log.info(df_dict)
        if df_agg.exists(df_dict):
            continue

        set_global_seeds(seed)
        init_alpha = np.random.uniform(low=0.5, high=1.0)
        log.info(f"init_alpha={init_alpha}")
        out = run_dro(
            data=data,
            optim_config=replace(
                optim_config,
                init_alpha=init_alpha,
                num_actions=num_actions,
                seed=seed,
                delta=delta,
                **learn_optim_config,
            ),
        )
        policy = out["policy"]
        train_obj = out["traj"][-1]["phi_n"]

        # Evaluate
        probs_mat = np.array(policy.probs(test_data.s))
        eval_res = get_eval_results_for_policy(
            data=replace(test_data, probs_mat=probs_mat),
            seed=seed,
        )
        df_dict.update(eval_res)
        df_dict["train_obj"] = train_obj

        df_agg.append(df_dict)
