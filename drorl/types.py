"""
Datasets have type DataInput.
When running run_dro in dro.py, the configuration is set using OptimConfig.
"""

import chex
import jax.numpy as jnp
from dataclasses import field
from enum import Enum

from typing import Optional, List


@chex.dataclass(frozen=True)
class DataInput:
    s: jnp.ndarray
    # target actions
    a: jnp.ndarray
    a_prob: jnp.ndarray
    r: jnp.ndarray

    # target policy distribution
    probs_mat: Optional[jnp.ndarray] = None

    # may be normalized
    ips_weights: Optional[jnp.ndarray] = None

    # for ground truth
    reward_mat: Optional[jnp.ndarray] = None

    # outcome fns: fi^*(s, a; alpha) = E[ r^i exp(-r/alpha) ]
    f0_mat: Optional[jnp.ndarray] = None
    f1_mat: Optional[jnp.ndarray] = None

    # cond_weights: num_actions x N/K x K x N-N/K (i.e. num_train_samples)
    # cond_weights[a,i,:] is estimate of distribution of r|s_i,a and should sum to 1.
    # so if r is the vector of training rewards (of shape num_train_samples),
    cond_weights: Optional[jnp.ndarray] = None
    # K x N-N/K.
    # 0: r[N/K:] (training rewards of first fold)
    # 1: r[:N/K], r[2N/K:]
    # ...
    r_for_rf: Optional[jnp.ndarray] = None

    # for debugging
    opt_a: Optional[jnp.ndarray] = None


class OptMethod(str, Enum):
    NEWTONS = "NEWTONS"
    GRADIENT_ASCENT = "GRADIENT_ASCENT"
    MULTIDIMENSIONAL_NEWTONS = "MULTIDIMENSIONAL_NEWTONS"


class FitPiMethod(str, Enum):
    XGBOOST = "XGBOOST"


class FitOutcomeFnMethod(str, Enum):
    GROUND_TRUTH = "GROUND_TRUTH"
    XGBOOST = "XGBOOST"
    RANDOM_FOREST_CONTINUUM = "RANDOM_FOREST_CONTINUUM"


class AlphaUpdateMethod(str, Enum):
    FULL_INFO = "FULL_INFO"
    IPS = "IPS"
    LDR = "LDR"


@chex.dataclass(frozen=True)
class OptimConfig:
    data_path: str = "__fill_me_in__"

    seed: int = 0
    converge_criterion: float = 5e-5
    num_consecutive_stops_needed: int = 5

    min_alpha_clip: float = 0.01
    max_alpha_clip: float = 1e2

    alpha_update_method: AlphaUpdateMethod = AlphaUpdateMethod.FULL_INFO
    self_normalize_ips: bool = True
    
    # uncertainty set radius
    delta: float = 0.1
    max_iters: int = 500
    init_alpha: float = 2.0
    num_actions: int = 3
    # if None, don't cross-fit. Otherwise, should be at least 2.
    crossfit_num_folds: Optional[int] = None

    fit_pihat_method: Optional[FitPiMethod] = None

    # Whether to run LDR^2OPE.
    run_ldml: bool = False
    num_inner_splits: int = 2
    fit_outcome_fn_method: Optional[FitOutcomeFnMethod] = None

    verbose: bool = False
    opt_method: OptMethod = OptMethod.NEWTONS

    # only relevant for gradient ascent
    opt_lr: float = 3.0

    ### DROPL ###
    learn_pi: bool = False
    update_pi_frequency: int = 1
    # if none, then optimize alpha to completion
    num_alpha_steps_between: Optional[int] = None

    warmstart_mlp: bool = True
    learn_mlp_hidden: List = field(default_factory=list)
    learn_mlp_temp: float = 0.1
    learn_mlp_epochs: int = 50
    learn_mlp_batch_size: int = 256
    learn_mlp_lr: float = 1e-2
    learn_mlp_optimizer: str = "sgd"

    # these perturbations are extra runs in training.
    learn_mlp_num_perturbations: int = 0
