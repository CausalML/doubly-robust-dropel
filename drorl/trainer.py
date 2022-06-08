"""
Define and optimize a neural softmax policy for a given loss function.
"""
import jax
import numpy as np
import jax.numpy as jnp
from utils import adv_idx
import haiku as hk
from typing import List
import optax
from utils import vectorized_multinomial
import copy
import logging

log = logging.getLogger(__name__)


def create_est_fn(est_method, a, a_prob, r, reward_mat=None, regress_mat=None):
    """
    Return the loss/reward function, based on which estimation method we're using.
    """
    a, a_prob, r = [jnp.array(x) for x in [a, a_prob, r]]
    reward_mat = jnp.array(reward_mat) if reward_mat is not None else None
    regress_mat = jnp.array(regress_mat) if regress_mat is not None else None

    if est_method == "full_info":

        @jax.jit
        def est_fn(idxs, probs):
            return (reward_mat[idxs] * probs).sum(1).mean()

        return est_fn

    if est_method == "ips":

        @jax.jit
        def est_fn(idxs, probs):
            ips = adv_idx(probs, a[idxs]) / a_prob[idxs]
            return (ips * r[idxs]).mean()

        return est_fn

    if est_method == "snips":

        @jax.jit
        def est_fn(idxs, probs):
            ips = adv_idx(probs, a[idxs]) / a_prob[idxs]
            ips = ips / ips.mean()
            return (ips * r[idxs]).mean()

        return est_fn

    if est_method == "dr":

        @jax.jit
        def est_fn(idxs, probs):
            ips = adv_idx(probs, a[idxs]) / a_prob[idxs]
            a_outcome = adv_idx(regress_mat[idxs], a[idxs])
            dr_vec = ips * (r[idxs] - a_outcome) + (regress_mat[idxs] * probs).sum(1)

            return dr_vec.mean()

        return est_fn

    raise NotImplementedError(est_method)


class SoftmaxPolicy:
    """
    MLP takes state and outputs a value for each action.
    """

    def __init__(self, mlp, mlp_params):
        self.mlp = mlp
        self.mlp_params = mlp_params

    def act(self, s):
        return vectorized_multinomial(self.probs(s))

    def probs(self, s):
        return self.mlp.apply(self.mlp_params, s)


optimizer_map = {
    "sgd": optax.sgd,
    "adam": optax.adam,
    "adagrad": optax.adagrad,
    "rmsprop": optax.rmsprop,
}


def perturb_weights(weights, rng_key):
    flatmap_cls = type(weights)

    def _perturb(flatmap):
        nonlocal rng_key

        d = copy.deepcopy(flatmap)
        # d = d._to_mapping()
        for k in d:
            v = d[k]
            if isinstance(v, jax.numpy.ndarray) or isinstance(v, np.ndarray):
                norm = jnp.linalg.norm(v)
                rng_key, subkey = jax.random.split(rng_key)
                perturbation = (norm / v.size) * jax.random.normal(subkey, shape=v.shape)

                d[k] = v + perturbation
            else:
                d[k] = _perturb(v)
        return flatmap_cls(d)
    return _perturb(weights)
            


class MLPTrainer:
    def __init__(
        self,
        state_dim: int,
        num_actions: int,
        mlp_hidden: List[int],
        temp: float,
        learning_rate: float,
        rng_key,
        verbose: bool = False,
        optimizer: str = "sgd",
    ):
        self.verbose = verbose

        ## Outputs probabilities for each action
        class MLP(hk.Module):
            def __init__(self):
                super().__init__(name="mlp")

            def __call__(self, inputs):
                logits = hk.nets.MLP(
                    output_sizes=[*mlp_hidden, num_actions],
                    activation=jax.nn.relu,
                    activate_final=False,
                    w_init=hk.initializers.Orthogonal(scale=1.0, axis=-1)
                )(inputs)
                return jax.nn.softmax(logits / temp, axis=-1)

        mlp_fn = lambda inputs: MLP()(inputs)
        self.state_dim = state_dim
        self.mlp = hk.without_apply_rng(hk.transform(mlp_fn))

        rng_key, *subkey = jax.random.split(rng_key, 3)
        
        assert optimizer in optimizer_map, f"invalid {optimizer}"
        self.optim = optax.chain(
            optax.clip(1.0), optimizer_map[optimizer](learning_rate)
        )

        mlp_params = self.mlp.init(rng=subkey[0], inputs=jax.random.normal(rng_key, shape=(1, self.state_dim)))
        opt_state = self.optim.init(mlp_params)
        self.train_state = {
            "mlp_params": mlp_params,
            "opt_state": opt_state,
            "rng_key": rng_key,
        }


    def learn(
        self,
        s,
        est_fn,
        max_epochs: int = 50,
        batch_size: int = 256,
        minimize: bool = False,
        num_extra_perturbations: int = 0,
    ):
        loss_multiplier = 1.0 if minimize else -1.0
        num_batches = s.shape[0] // batch_size

        @jax.jit
        def update_fn(train_state, s, idxs):
            def loss_fn(mlp_params):
                probs = self.mlp.apply(mlp_params, inputs=s)
                return loss_multiplier * est_fn(idxs, probs)

            loss, grad = jax.value_and_grad(loss_fn)(train_state["mlp_params"])
            updates, opt_state = self.optim.update(
                grad, train_state["opt_state"], train_state["mlp_params"]
            )
            new_mlp_params = optax.apply_updates(train_state["mlp_params"], updates)
            train_state.update(
                {
                    "mlp_params": new_mlp_params,
                    "opt_state": opt_state,
                }
            )
            return train_state, loss

        self.update_fn = update_fn

        ### add some perturbations, don't perturb the first one.
        rng_key = self.train_state["rng_key"]
        starting_params = copy.deepcopy(self.train_state)
        final_losses = []
        best_loss = None
        best_params = None
        for p in range(num_extra_perturbations+1):
            if p > 0:
                rng_key, subkey = jax.random.split(rng_key)
                self.train_state["mlp_params"] = perturb_weights(starting_params["mlp_params"], subkey)
                self.train_state["opt_state"] = starting_params["opt_state"]

            for e in range(max_epochs):
                rng_key, subkey = jax.random.split(rng_key)
                shuffled_idxs = jax.random.permutation(subkey, s.shape[0])
                epoch_loss = 0.0
                for i in range(num_batches):
                    idxs = shuffled_idxs[i * batch_size : (i + 1) * batch_size]
                    self.train_state, loss = self.update_fn(
                        train_state=self.train_state,
                        s=s[idxs],
                        idxs=idxs,
                    )
                    epoch_loss += loss * idxs.shape[0]
                epoch_loss = epoch_loss / (num_batches * batch_size)

            final_losses.append(epoch_loss.item())
            if best_loss is None or epoch_loss < best_loss:
                best_loss = epoch_loss
                best_params = copy.deepcopy(self.train_state)

        self.train_state = best_params
        self.train_state["rng_key"] = rng_key
        policy = SoftmaxPolicy(self.mlp, self.train_state["mlp_params"])
        return policy
