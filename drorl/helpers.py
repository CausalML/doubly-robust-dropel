import jax
import numpy as np
import jax.numpy as jnp
from sklearn.ensemble import RandomForestRegressor
from drorl.types import DataInput
from dataclasses import fields

## XGBoost doesn't work on M1 chip
# import xgboost
import lightgbm as lgb

TREE_MAX_DEPTH = 3
NUM_JOBS = 2


def split_data(v, num_splits: int, remove_idx: int):
    """
    If not evenly splittable, give the first few the most.
    Ex. when splitting 10 into 3, give 4, 3, 3
    When splitting 11 into 3, make 4, 4, 3
    """
    assert 0 <= remove_idx < num_splits
    n = v.shape[0]
    num_per_split = n // num_splits
    remainder = n % num_splits
    start_idx = num_per_split * remove_idx + min(remainder, remove_idx)
    end_idx = num_per_split * (remove_idx + 1) + min(remainder, remove_idx + 1)
    assert v.shape[0] == n
    before = v[:start_idx]
    fold = v[start_idx:end_idx]
    after = v[end_idx:]
    return (np.concatenate((before, after)), fold)


# equivalent to calling split_data on all fields
def split_data_input(data_input: DataInput, num_splits: int, remove_idx: int):
    remaining = {}
    removed = {}
    for f in fields(data_input):
        v = getattr(data_input, f.name)
        if v is not None:
            v_remaining, v_removed = split_data(
                v, num_splits=num_splits, remove_idx=remove_idx
            )
            remaining[f.name] = v_remaining
            removed[f.name] = v_removed
    return DataInput(**remaining), DataInput(**removed)


class Predictor:
    def __init__(self):
        pass

    def fit(self, x, y):
        pass

    def update(self, **kwargs):
        # update after being fitted
        pass

    def __call__(self, *args, **kwargs):
        pass


class PropensityPredictor(Predictor):
    def __init__(self, num_actions: int, seed: int):
        assert num_actions >= 2
        self.reg = lgb.LGBMClassifier(
            max_depth=TREE_MAX_DEPTH,
            n_jobs=2,
            random_state=seed,
            objective="binary" if num_actions == 2 else "multiclass",
            num_classes=num_actions,
        )

    def fit(self, state, action):
        self.reg.fit(state, action)

    def update(self):
        raise NotImplementedError()

    def __call__(self, state, action):
        prob_matrix = self.reg.predict_proba(state).astype(np.float32)
        return prob_matrix[jnp.arange(action.shape[0]), action]


# @jax.jit
def _concat_state_action(state: jnp.ndarray, action: jnp.ndarray, num_classes: int):
    return jnp.concatenate(
        (state, jax.nn.one_hot(action, num_classes=num_classes)), axis=1
    )


class RegressorPredictor(Predictor):
    def __init__(self, num_actions: int, seed: int):
        self.num_actions = num_actions
        self.reg = lgb.LGBMRegressor(
            max_depth=TREE_MAX_DEPTH,
            n_jobs=2,
            random_state=seed,
        )

    def fit(self, state, action, target):
        x = _concat_state_action(state, action, self.num_actions)
        self.reg.fit(x, target)

    def update(self):
        raise NotImplementedError()

    def __call__(self, state, action):
        x = _concat_state_action(state, action, self.num_actions)
        return self.reg.predict(x).astype(np.float32)


@jax.jit
def calc_cond_weights(train_ids, test_ids):
    test_ids = jnp.expand_dims(test_ids, axis=1)
    P = train_ids == test_ids
    # num_test_samples x num_training
    weights = (P.astype(np.float32) / P.sum(1, keepdims=True)).mean(2)
    return weights


class ContinuumRegressorPredictor(Predictor):
    """
    Uses random forest to get the weights.
    """

    def __init__(self, num_actions: int, seed: int):
        self.num_actions = num_actions
        self.alpha = None
        self.rf = RandomForestRegressor(
            n_estimators=25,
            min_samples_leaf=5,
            n_jobs=4,
            random_state=seed,
        )

    def fit(self, state, action, reward):
        x = _concat_state_action(state, action, self.num_actions)
        self.rf.fit(x, reward)
        self.train_ids = self.rf.apply(x)
        self.train_reward = reward

    def __call__(self, state, action):
        x = _concat_state_action(state, action, self.num_actions)
        test_ids = self.rf.apply(x)
        weights = calc_cond_weights(self.train_ids, test_ids)
        return {
            "weights": weights,
            "train_reward": self.train_reward,
        }


class XFitWrapper:
    """
    Assume fit_fn: (x, y) -> predictor
    And predictor(x) gives outcome (whether it be predicted probabilities, or regression)
    """

    def __init__(self, K: int, predictor_constructor, concat_result: bool = True):
        self.K = K
        self.predictor_constructor = predictor_constructor
        self.xfit_predictors = None
        self.concat_result = concat_result

    def fit(self, **kwargs):
        """
        Returns both the xfitted_predictors and training_data_predictions
        """
        self.xfit_predictors = {}
        for k in range(self.K):
            split_kwargs = {}
            for key, val in kwargs.items():
                split_kwargs[key], _ = split_data(val, num_splits=self.K, remove_idx=k)
            self.xfit_predictors[k] = self.predictor_constructor()
            self.xfit_predictors[k].fit(**split_kwargs)

    def __call__(self, **kwargs):
        outs = []
        for k in range(self.K):
            split_kwargs = {}
            for key, val in kwargs.items():
                _, split_kwargs[key] = split_data(val, num_splits=self.K, remove_idx=k)
            out = self.xfit_predictors[k](**split_kwargs)
            outs.append(out)

        if self.concat_result:
            return jnp.concatenate(outs, axis=0)
        else:
            return outs
