"""
This file contains the core algorithms for DROPE/DROPL. 
The entry point is the function `run_dro`.
If optim_config.learn_pi == False, then it runs DROPE. 
If optim_config.learn_pi == True, then it runs DROPL. 
Check out types.py for more information on training parameters.


For DROPE, the only job of `run_dro` is to optimize alpha. 
`full_information`, `ips` and `ldml` estimate several key parameters,
which are passed into `_get_W_terms`, `_get_DR_reward_terms` and `_scalar_update`
to perform one-step update for optimizing alpha.

For DROPL, `run_dro` must also optimize a policy concurrently.
This is done by instantiating a trainer from trainer.py.
"""
from dataclasses import replace
import jax.numpy as jnp
import numpy as np

from utils import *
from drorl.types import (
    DataInput,
    FitPiMethod,
    OptimConfig,
    OptMethod,
    FitOutcomeFnMethod,
    AlphaUpdateMethod,
)
from drorl.trainer import create_est_fn, MLPTrainer
from drorl.helpers import (
    ContinuumRegressorPredictor,
    RegressorPredictor,
    XFitWrapper,
    split_data_input,
    PropensityPredictor,
)
import logging
import jax
from tqdm import tqdm
import math

log = logging.getLogger(__name__)


class TrainingFailureException(Exception):
    pass


EPS = 1e-6


@jax.jit
def _get_W_terms(
    r: np.ndarray,
    alpha: float,
    ipw: np.ndarray,
):
    assert ipw is not None
    W_ik = ipw * jnp.exp(-r / alpha)
    W_nk = jnp.mean(W_ik)

    dW_ik = ipw * r * jnp.exp(-r / alpha)
    # since LDML scaled it by alpha^2, we'll do the same here...
    dW_nk = jnp.mean(dW_ik)
    d2W_ik = (
        ipw * jnp.exp(-r / alpha) * ((r ** 2) / (alpha ** 4) - (2 * r) / (alpha ** 3))
    )
    d2W_nk = jnp.mean(d2W_ik)
    return W_nk, dW_nk, d2W_nk


@jax.jit
def _get_DR_reward_terms(
    W_n: float, dW_n: float, d2W_n: float, alpha: float, delta: float
):
    # since LDML scaled it by alpha^2
    dW_n = dW_n / (alpha ** 2)
    DR_reward = -alpha * jnp.log(W_n) - alpha * delta
    d_DR_reward = -jnp.log(W_n) - delta - alpha * dW_n / W_n
    d2_DR_reward = -2.0 * dW_n / W_n - alpha * (d2W_n * W_n - dW_n ** 2) / (W_n ** 2)
    return DR_reward, d_DR_reward, d2_DR_reward


# @jax.jit
def _scalar_update(
    alpha: float,
    dphi_n: float,
    d2phi_n: float,
    optim_config: OptimConfig,
    step: int,
):
    if optim_config.opt_method == OptMethod.NEWTONS:
        assert abs(d2phi_n) > EPS, f"Invalid {d2phi_n}.."
        alpha = alpha - dphi_n / d2phi_n
    elif optim_config.opt_method == OptMethod.GRADIENT_ASCENT:
        alpha = alpha + optim_config.opt_lr * dphi_n / (min(step, 100) ** 0.5)
    else:
        raise RuntimeError()

    return float(alpha)


@jax.jit
def full_information(alpha: float, data: DataInput, delta: float):
    assert data.probs_mat.shape == data.reward_mat.shape

    W_ik = jnp.exp(-data.reward_mat / alpha)
    W_n = jnp.mean((W_ik * data.probs_mat).sum(1))

    dW_ik = data.reward_mat * jnp.exp(-data.reward_mat / alpha)
    # since LDML scaled it by alpha^2, we'll do the same here...
    dW_n = jnp.mean((dW_ik * data.probs_mat).sum(1))
    d2W_ik = jnp.exp(-data.reward_mat / alpha) * (
        (data.reward_mat ** 2) / (alpha ** 4) - (2 * data.reward_mat) / (alpha ** 3)
    )
    d2W_n = jnp.mean((d2W_ik * data.probs_mat).sum(1))
    phi_n, dphi_n, d2phi_n = _get_DR_reward_terms(
        W_n=W_n, dW_n=dW_n, d2W_n=d2W_n, alpha=alpha, delta=delta
    )
    return {
        "W_n": W_n,
        "dW_n": dW_n,
        "d2W_n": d2W_n,
        "phi_n": phi_n,
        "dphi_n": dphi_n,
        "d2phi_n": d2phi_n,
    }


@jax.jit
def get_ips_weights(a, probs_mat, a_prob):
    target_a_prob = adv_idx(probs_mat, a)
    ips = target_a_prob / a_prob
    return ips


@jax.jit
def get_snips_weights(a, probs_mat, a_prob):
    target_a_prob = adv_idx(probs_mat, a)
    ips = target_a_prob / a_prob
    return ips / jnp.mean(ips)


@jax.jit
def ips(alpha: float, data: DataInput, delta: float):
    ips = data.ips_weights
    W_n, dW_n, d2W_n = _get_W_terms(r=data.r, alpha=alpha, ipw=ips)
    phi_n, dphi_n, d2phi_n = _get_DR_reward_terms(
        W_n=W_n, dW_n=dW_n, d2W_n=d2W_n, alpha=alpha, delta=delta
    )
    return {
        "W_n": W_n,
        "dW_n": dW_n,
        "d2W_n": d2W_n,
        "phi_n": phi_n,
        "dphi_n": dphi_n,
        "d2phi_n": d2phi_n,
    }


@jax.jit
def ldml(alpha: float, data: DataInput, delta: float):
    ips = data.ips_weights
    exp_term = jnp.exp(-data.r / alpha)

    W_n_IPW = jnp.mean(ips * exp_term)
    W_n_DR = jnp.mean(
        -ips * adv_idx(data.f0_mat, data.a) + (data.probs_mat * data.f0_mat).sum(1)
    )
    W_n = W_n_IPW + W_n_DR

    dW_n_IPW = jnp.mean(ips * data.r * exp_term)
    dW_n_DR = jnp.mean(
        -ips * adv_idx(data.f1_mat, data.a) + (data.probs_mat * data.f1_mat).sum(1)
    )
    dW_n = dW_n_IPW + dW_n_DR

    J_11 = jnp.mean(ips * exp_term * data.r) / (alpha ** 2)
    J_21 = jnp.mean(ips * exp_term * (data.r ** 2)) / (alpha ** 2)

    # solve W_n, dW_n, phi_n directly
    phi_n = -alpha * jnp.log(W_n) - alpha * delta
    g = -delta - jnp.log(W_n) - dW_n / (alpha * W_n)
    g_grad = -J_11 / W_n - (J_21 * alpha * W_n - dW_n * (W_n * alpha * J_11)) / (
        alpha * alpha * W_n * W_n
    )
    return {
        "W_n_IPW": W_n_IPW,
        "W_n_DR": W_n_DR,
        "W_n": W_n,
        "dW_n_IPW": dW_n_IPW,
        "dW_n_DR": dW_n_DR,
        "dW_n": dW_n,
        "J_11": J_11,
        "J_21": J_21,
        "phi_n": phi_n,
        "dphi_n": g,
        "d2phi_n": g_grad,
    }


def _get_regression_mat(data: DataInput, regressor, num_actions: int):
    outs = []
    for a in range(num_actions):
        a = np.full(shape=(data.s.shape[0],), fill_value=a, dtype=np.int32)
        out = regressor(state=data.s, action=a)
        outs.append(out)
    if isinstance(outs[0], jnp.ndarray) or isinstance(outs[0], np.ndarray):
        return jnp.stack(outs, axis=1)
    elif isinstance(outs[0], dict):
        assert False
    else:
        raise NotImplementedError(
            f"Invalid out type {type(outs[0])}. Outs[0] = {outs[0]}"
        )


def _localization_step(data: DataInput, optim_config: OptimConfig):
    alpha_list = []
    f0_list = []
    f1_list = []
    for k in range(optim_config.crossfit_num_folds):
        # partition data into h1, h2, test; train on h1, h2 and predict on test
        xfit_data, test_data = split_data_input(
            data, num_splits=optim_config.crossfit_num_folds, remove_idx=k
        )
        num_inner_successes = 0
        inner_f0 = np.zeros(shape=(test_data.s.shape[0], optim_config.num_actions))
        inner_f1 = np.zeros(shape=(test_data.s.shape[0], optim_config.num_actions))
        for inner_k in range(optim_config.num_inner_splits):
            h1_data, h2_data = split_data_input(
                xfit_data,
                num_splits=optim_config.num_inner_splits,
                remove_idx=inner_k,
            )

            # Get alpha_init from running subroutine on h1
            # If optim_config.fit_outcome_every_step is on, we fit outcome for every alpha along trajectory
            for subroutine_opt_method in [
                OptMethod.GRADIENT_ASCENT,
                OptMethod.NEWTONS,
            ]:
                subroutine_optim_config = replace(
                    optim_config,
                    crossfit_num_folds=(optim_config.crossfit_num_folds - 1)
                    // optim_config.num_inner_splits,
                    opt_method=subroutine_opt_method,
                    alpha_update_method=AlphaUpdateMethod.IPS,
                    self_normalize_ips=True,
                    verbose=False,
                )
                subroutine_out = run_alpha_updates(
                    alpha=optim_config.init_alpha,
                    data=h1_data,
                    optim_config=subroutine_optim_config,
                    show_pbar=False,
                )
                if subroutine_out is not None:
                    break

            ## NOTE: if process fails, we simply ignore it:
            # alpha_init is undefined (and not included for averaging)
            # and f0/f1 estimates are 0
            if subroutine_out is not None:
                alpha_init_k = subroutine_out["alpha"]
                assert alpha_init_k > 0, f"Invalid {alpha_init_k}"
                del h1_data

                if (
                    optim_config.fit_outcome_fn_method
                    == FitOutcomeFnMethod.GROUND_TRUTH
                ):
                    f0hat_k = np.exp(-test_data.reward_mat / alpha_init_k)
                    f1hat_k = test_data.reward_mat * np.exp(
                        -test_data.reward_mat / alpha_init_k
                    )
                elif optim_config.fit_outcome_fn_method == FitOutcomeFnMethod.XGBOOST:
                    regressor_constructor = lambda: RegressorPredictor(
                        num_actions=optim_config.num_actions, seed=optim_config.seed
                    )
                    f0_trainer = regressor_constructor()
                    f0_trainer.fit(
                        state=h2_data.s,
                        action=h2_data.a,
                        target=np.exp(-h2_data.r / alpha_init_k),
                    )
                    f0hat_k = _get_regression_mat(
                        test_data, f0_trainer, optim_config.num_actions
                    )

                    f1_trainer = regressor_constructor()
                    f1_trainer.fit(
                        state=h2_data.s,
                        action=h2_data.a,
                        target=h2_data.r * np.exp(-h2_data.r / alpha_init_k),
                    )
                    f1hat_k = _get_regression_mat(
                        test_data, f1_trainer, optim_config.num_actions
                    )
                else:
                    raise NotImplementedError()

                inner_f0 = inner_f0 + f0hat_k
                inner_f1 = inner_f1 + f1hat_k
                num_inner_successes += 1
                alpha_list.append(alpha_init_k)

        inner_f0 = inner_f0 / num_inner_successes
        inner_f1 = inner_f1 / num_inner_successes
        f0_list.append(inner_f0)
        f1_list.append(inner_f1)

    data = replace(
        data,
        f0_mat=np.concatenate(f0_list, axis=0),
        f1_mat=np.concatenate(f1_list, axis=0),
    )
    # update alpha to be localized version
    alpha = np.mean(np.array(alpha_list))
    return data, alpha


@jax.jit
def calc_fi_from_weights(cond_weights, r_mat, alpha, i: int):
    """
    cond_weights: num_actions x N/K x K x (N - N/K)
    r_mat: K x (N - N/K)
    Also works for non-crossfit version, where cond_weights: num_actions x N x N, r_mat: N
    Sums across training examples axis.
    """
    fi_expr = (r_mat ** i) * jnp.exp(-r_mat / alpha)
    weighted_est = jnp.sum(cond_weights * fi_expr, axis=-1)
    out = jnp.reshape(weighted_est, (cond_weights.shape[0], -1), order="F")
    out = jnp.transpose(out, axes=(1, 0))
    return out


def run_alpha_updates(
    alpha: float, data: DataInput, optim_config: OptimConfig, show_pbar: bool = False
):
    has_converged = False
    stop_count = 0
    num_projections_so_far = 0

    state_dict = {}
    new_alpha = alpha

    # calculate ips weights
    if optim_config.alpha_update_method in [
        AlphaUpdateMethod.IPS,
        AlphaUpdateMethod.LDR,
    ]:
        if optim_config.self_normalize_ips:
            ips_weights = get_snips_weights(data.a, data.probs_mat, data.a_prob)
        else:
            ips_weights = get_ips_weights(data.a, data.probs_mat, data.a_prob)
        data = replace(data, ips_weights=ips_weights)

    max_iters = optim_config.max_iters
    if optim_config.num_alpha_steps_between:
        max_iters = optim_config.num_alpha_steps_between

    pbar = range(1, max_iters + 1)
    if show_pbar:
        pbar = tqdm(pbar)
    for t in pbar:
        alpha = new_alpha
        if (
            optim_config.fit_outcome_fn_method
            == FitOutcomeFnMethod.RANDOM_FOREST_CONTINUUM
        ):
            f0_regress_mat = calc_fi_from_weights(
                data.cond_weights, data.r_for_rf, alpha, i=0
            )
            f1_regress_mat = calc_fi_from_weights(
                data.cond_weights, data.r_for_rf, alpha, i=1
            )
            data = replace(data, f0_mat=f0_regress_mat, f1_mat=f1_regress_mat)

        # update alpha for fixed policy
        try:
            if optim_config.alpha_update_method == AlphaUpdateMethod.FULL_INFO:
                new_state_dict = full_information(alpha, data, optim_config.delta)
            elif optim_config.alpha_update_method == AlphaUpdateMethod.IPS:
                new_state_dict = ips(alpha, data, optim_config.delta)
            elif optim_config.alpha_update_method == AlphaUpdateMethod.LDR:
                new_state_dict = ldml(alpha, data, optim_config.delta)
            else:
                raise NotImplementedError(
                    f"alpha_update_method={optim_config.alpha_update_method}"
                )

        except TrainingFailureException:
            return None

        # since np array isn't encodable
        for k, v in new_state_dict.items():
            try:
                new_state_dict[k] = v.item()
            except:
                pass

        d2phi_n = new_state_dict["d2phi_n"]
        if optim_config.opt_method == OptMethod.NEWTONS and (
            math.isnan(d2phi_n) or abs(d2phi_n) <= EPS
        ):
            log.info(
                f"Breaking out to prevent invalid Newton's step (nan or divide by 0). d2phi_n: {d2phi_n}"
            )
            log.info("state_dict: ", new_state_dict)
            log.info("alpha: ", alpha)
            return None

        new_alpha = _scalar_update(
            alpha,
            optim_config=optim_config,
            dphi_n=new_state_dict["dphi_n"],
            d2phi_n=d2phi_n,
            step=t,
        )

        if (
            math.isnan(new_alpha)
            or new_alpha > optim_config.max_alpha_clip
            or new_alpha < optim_config.min_alpha_clip
        ):
            log.info(
                f"Reached invalid alpha: {new_alpha} (either nan or outside [{optim_config.min_alpha_clip}, {optim_config.max_alpha_clip}]"
            )
            log.info(f"State_dict: {new_state_dict}")
            rand_proj = random.uniform(
                optim_config.min_alpha_clip, optim_config.max_alpha_clip
            )
            log.info(f"Projecting to {rand_proj}")
            new_alpha = rand_proj
            num_projections_so_far += 1
            if num_projections_so_far > 10:
                log.info("Constantly bad alpha. Stopping now...")
                return None

        if optim_config.verbose:
            log.info("*" * 20 + f"Step {t}" + "*" * 20)
            for k, v in new_state_dict.items():
                if isinstance(v, float):
                    old_v = state_dict.get(k, None)
                    direction = None if old_v is None else "up" if v > old_v else "down"
                    old_v_str = f"{old_v:.3f}" if isinstance(old_v, float) else "None"
                    log.info(f"{k} \t\t {old_v_str} -> {v:.3f} \t\t ({direction})")

        # technically this will test t+1 vs. t for alpha and t vs. t-1 for others
        norm = 0.0
        theta = ["phi_n", "W_n", "dW_n"]
        for k in theta:
            norm += (state_dict.get(k, np.inf) - new_state_dict[k]) ** 2
        norm += (alpha - new_alpha) ** 2
        norm = norm ** 0.5
        if show_pbar:
            pbar.set_postfix(diff_norm=norm, stop_count=stop_count)

        if norm < optim_config.converge_criterion:
            stop_count += 1
            if stop_count >= optim_config.num_consecutive_stops_needed:
                has_converged = True
                break
        else:
            stop_count = 0

        state_dict = new_state_dict

    new_state_dict["converged"] = has_converged
    new_state_dict["alpha"] = alpha
    new_state_dict["seed"] = optim_config.seed
    return new_state_dict


def _fit_pihat(data, optim_config):
    predictor_constructor = lambda: PropensityPredictor(
        num_actions=optim_config.num_actions, seed=optim_config.seed
    )
    if optim_config.crossfit_num_folds is None:
        pihat_trainer = predictor_constructor()
    else:
        pihat_trainer = XFitWrapper(
            K=optim_config.crossfit_num_folds,
            predictor_constructor=predictor_constructor,
        )
    pihat_trainer.fit(state=data.s, action=data.a)
    pihat = pihat_trainer(state=data.s, action=data.a)
    data = replace(data, a_prob=pihat)
    log.info("Finished fitting pihat!")
    return data


def _fit_continuum_weights(data, optim_config):
    predictor_constructor = lambda: ContinuumRegressorPredictor(
        num_actions=optim_config.num_actions, seed=optim_config.seed
    )
    if optim_config.crossfit_num_folds is None:
        continuum_outcome_regressor = predictor_constructor()
    else:
        ## In this case, require N/K to be whole number.
        assert data.s.shape[0] % optim_config.crossfit_num_folds == 0
        continuum_outcome_regressor = XFitWrapper(
            K=optim_config.crossfit_num_folds,
            predictor_constructor=predictor_constructor,
            concat_result=False,
        )
    continuum_outcome_regressor.fit(state=data.s, action=data.a, reward=data.r)
    log.info("Finished fitting continuum predictor!")

    # reward should be same for all actions
    r_for_rf = None
    action_weight_list = []
    for a in range(optim_config.num_actions):
        log.info(f"Finished getting action {a}")
        out = continuum_outcome_regressor(state=data.s, action=jnp.full_like(data.a, a))
        weight_mat = None
        if isinstance(out, list):
            # shape N/K x K x N-N/K
            weight_mat = jnp.stack([o["weights"] for o in out], axis=1)
            if r_for_rf is None:
                # shape K x N-N/K
                r_for_rf = jnp.stack([o["train_reward"] for o in out], axis=0)
        else:
            # shape N x N
            weight_mat = out["weights"]
            if r_for_rf is None:
                r_for_rf = out["train_reward"]
        action_weight_list.append(weight_mat)

    cond_weights = jnp.stack(action_weight_list, axis=0)
    data = replace(data, cond_weights=cond_weights, r_for_rf=r_for_rf)
    log.info("Finished getting continuum weights!")
    return data


def _policy_learning_step(
    alpha: float,
    data: DataInput,
    optim_config: OptimConfig,
    trainer,
):
    est_method = None
    reward_mat = None
    regress_mat = None
    if optim_config.alpha_update_method == AlphaUpdateMethod.FULL_INFO:
        est_method = "full_info"
        reward_mat = jnp.exp(-data.reward_mat / alpha)
    elif optim_config.alpha_update_method == AlphaUpdateMethod.IPS:
        est_method = "ips"
    elif optim_config.alpha_update_method == AlphaUpdateMethod.LDR:
        assert (
            optim_config.fit_outcome_fn_method
            == FitOutcomeFnMethod.RANDOM_FOREST_CONTINUUM
        )
        est_method = "dr"
        # need to update regression since alpha changed
        f0_regress_mat = calc_fi_from_weights(
            data.cond_weights, data.r_for_rf, alpha, i=0
        )
        f1_regress_mat = calc_fi_from_weights(
            data.cond_weights, data.r_for_rf, alpha, i=1
        )
        data = replace(
            data,
            f0_mat=f0_regress_mat,
            f1_mat=f1_regress_mat,
        )
        regress_mat = data.f0_mat
    else:
        raise NotImplementedError()

    est_fn = create_est_fn(
        est_method,
        a=data.a,
        a_prob=data.a_prob,
        r=jnp.exp(-data.r / alpha),
        reward_mat=reward_mat,
        regress_mat=regress_mat,
    )
    policy = trainer.learn(
        s=data.s,
        est_fn=est_fn,
        max_epochs=optim_config.learn_mlp_epochs,
        batch_size=optim_config.learn_mlp_batch_size,
        minimize=True,
        num_extra_perturbations=optim_config.learn_mlp_num_perturbations,
    )
    data = replace(data, probs_mat=policy.probs(data.s))
    return policy, data


def run_dro(data: DataInput, optim_config: OptimConfig):
    rng_key = set_global_seeds(optim_config.seed)

    # Fit the propensities
    if optim_config.fit_pihat_method is not None:
        data = _fit_pihat(data, optim_config)

    # DR with continuum of nuisances
    if optim_config.fit_outcome_fn_method == FitOutcomeFnMethod.RANDOM_FOREST_CONTINUUM:
        data = _fit_continuum_weights(data, optim_config)

    ## Only Evaluation (equiv. to not learning pi)
    alpha = optim_config.init_alpha
    if not optim_config.learn_pi:
        if optim_config.run_ldml:
            data, alpha = _localization_step(data, optim_config)
        return run_alpha_updates(alpha, data, optim_config, show_pbar=False)

    trainer = MLPTrainer(
        state_dim=data.s.shape[1],
        num_actions=optim_config.num_actions,
        mlp_hidden=optim_config.learn_mlp_hidden,
        temp=optim_config.learn_mlp_temp,
        rng_key=rng_key,
        learning_rate=optim_config.learn_mlp_lr,
        optimizer=optim_config.learn_mlp_optimizer,
    )
    if optim_config.warmstart_mlp:
        # Warmstart from a reasonable non-robust policy
        est_fn = create_est_fn("ips", a=data.a, a_prob=data.a_prob, r=data.r)
        policy = trainer.learn(s=data.s, est_fn=est_fn, max_epochs=30)
        data = replace(data, probs_mat=policy.probs(data.s))
        log.info("Finished warmstart!")

    val_list = []
    robust_val = np.inf
    stop_count = 0
    pbar = range(1, optim_config.max_iters + 1)
    pbar = tqdm(pbar)
    for t in pbar:

        ## First, take some policy update steps
        policy, data = _policy_learning_step(
            alpha,
            data,
            optim_config,
            trainer=trainer,
        )

        alpha_steps_out = run_alpha_updates(
            alpha,
            data,
            optim_config,
            show_pbar=False,
        )
        assert alpha_steps_out is not None

        new_alpha = alpha_steps_out["alpha"]
        new_robust_val = alpha_steps_out["phi_n"]
        phi_n_diff = abs(new_robust_val - robust_val)
        pbar.set_postfix(
            alpha_diff=abs(new_alpha - alpha),
            phi_n_diff=abs(new_robust_val - robust_val),
            stop_count=stop_count,
            alpha=new_alpha,
            phi_n=new_robust_val,
        )

        # also allow for when phi_n is super stable.
        has_converged = phi_n_diff < optim_config.converge_criterion

        if has_converged:
            stop_count += 1
            if stop_count >= optim_config.num_consecutive_stops_needed:
                break
        else:
            stop_count = 0

        alpha = new_alpha
        robust_val = new_robust_val
        val_list.append(
            {
                "alpha": alpha,
                "phi_n": robust_val,
            }
        )

    alpha_steps_out["policy"] = policy
    alpha_steps_out["seed"] = optim_config.seed
    alpha_steps_out["traj"] = val_list
    return alpha_steps_out
