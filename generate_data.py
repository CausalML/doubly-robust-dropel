"""
Run this file to generate datasets for DROPE/L experiments.
"""
from drorl.types import DataInput
import numpy as np
from typing import List
import jax
from utils import vectorized_multinomial
import pickle
import math, os
from utils import set_global_seeds, subarray_datacls
from drorl.types import AlphaUpdateMethod, OptimConfig, OptMethod
from drorl.dro import run_dro
import logging

log = logging.getLogger(__name__)


class ToyDataset:
    """Linear example from Si et al., 2020."""

    def __init__(self, betas: List[List[float]], sigmas: List[float]):
        self.betas = np.array(list([np.array(beta) for beta in betas])).astype(
            np.float32
        )
        self.sigmas = np.array(sigmas)
        assert self.betas.shape[0] == len(self.sigmas)
        self.state_dim = self.betas.shape[1]
        self.num_actions = self.betas.shape[0]

    def generate_data(self, n):
        s = np.random.uniform(low=-1.0, high=1.0, size=(n, self.state_dim))
        s_cross_beta = s @ self.betas.T
        opt_a = np.argmax(s_cross_beta, axis=1)
        means = s_cross_beta
        scales = self.sigmas
        reward_mat = np.random.normal(means, scale=scales)

        # Softmax action
        temp = 0.5
        logits = s_cross_beta
        a_prob_mat = np.array(jax.nn.softmax(logits / temp, axis=-1))
        a = vectorized_multinomial(a_prob_mat)
        a_prob = a_prob_mat[range(n), a]
        r = reward_mat[range(n), a]

        ## Target policy is softmax with temp 1.0
        target_a_prob_mat = np.array(jax.nn.softmax(logits, axis=-1))
        return DataInput(
            s=s,
            a=a,
            r=r,
            a_prob=a_prob,
            reward_mat=reward_mat,
            opt_a=opt_a,
            probs_mat=target_a_prob_mat,
        )


def save_dataset(
    data,
    dataset_name,
    seed: int,
    datasize_list: list,
    test_datasize: int,
    num_actions: int,
    **kwargs,
):
    ## Require action's to be {0, 1, ..., num_actions-1}
    ## Should contain at least one of each action.
    num_actions = data.a.max() + 1
    unique_a = np.unique(data.a)
    assert unique_a.shape[0] == num_actions
    for i in range(num_actions):
        assert unique_a[i] == i

    meta_dict = {
        "num_actions": num_actions,
        "training_sizes": datasize_list,
        "test_size": test_datasize,
        "deltas": deltas,
        "seed": seed,
        "ground_truth": {},
        **kwargs,
    }

    data_dir = f"data/{dataset_name}/{seed}/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir, exist_ok=True)

    assert datasize_list[-1] + test_datasize < data.s.shape[0]
    for datasize in datasize_list:
        data_file_path = f"data/{dataset_name}/{seed}/{datasize}.pkl"
        with open(data_file_path, "wb") as f:
            sub_data = subarray_datacls(data, datasize)
            pickle.dump(sub_data, f)

    test_data_path = f"data/{dataset_name}/{seed}/test.pkl"
    with open(test_data_path, "wb") as f:
        test_data = subarray_datacls(data, test_datasize, from_back=True)
        pickle.dump(test_data, f)

    ## Create meta dataset
    meta_file_path = f"data/{dataset_name}/{seed}/meta.pkl"
    if data.probs_mat is not None and len(data.probs_mat) > 0:
        for delta in deltas:
            optim_config = OptimConfig(
                seed=seed,
                delta=delta,
                num_actions=num_actions,
                alpha_update_method=AlphaUpdateMethod.FULL_INFO,
                opt_method=OptMethod.GRADIENT_ASCENT,
                # high bar,
                converge_criterion=1e-6,
                num_consecutive_stops_needed=20,
            )
            out = run_dro(
                data=data,
                optim_config=optim_config,
            )
            assert out is not None
            log.info(f"For {delta} seed {seed}, ground truth is: ", out)
            meta_dict["ground_truth"][delta] = out
    with open(meta_file_path, "wb") as f:
        pickle.dump(meta_dict, f)


if __name__ == "__main__":

    num_seeds = 30
    total_datapoints = 100000
    datasize_list = [
        1000,
        2000,
        3000,
        4000,
        5000,
        8000,
        10000,
        15000,
        20000,
    ]
    test_datasize = 20000

    deltas = [
        0.1,
        0.2,
        0.3,
        0.5,
    ]

    num_actions = 5
    dataset_name = f"linear_{num_actions}"
    a = np.array([i * 1j for i in range(num_actions)])

    ## Roots of unity, exp(2kpi i/N) for k=1, ..., N
    rou = np.exp(2 * math.pi * a / num_actions)
    betas = [[np.real(x), np.imag(x)] for x in rou]

    sigmas = [
        # Need this diversity to illustrate Distributional Robustness
        0.1 * k
        for k in range(1, num_actions + 1)
    ]

    for seed in range(num_seeds):
        set_global_seeds(seed)
        dataset = ToyDataset(betas=betas, sigmas=sigmas)
        data = dataset.generate_data(total_datapoints)
        save_dataset(
            data=data,
            dataset_name=dataset_name,
            seed=seed,
            datasize_list=datasize_list,
            test_datasize=test_datasize,
            num_actions=num_actions,
            betas=betas,
            sigmas=sigmas,
        )
