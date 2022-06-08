# Doubly Robust Distributionally Robust OPE/L

This repo contains code for the paper "Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning". \
The core implementation of our proposed Distributionally Robust OPE/L (DROPE/L) algorithms are in the folder `drorl`. Specifically,
* `drorl/dro.py` contains all code for performing DROPE and DROPL. The entry point is `run_dro`, which can be used for DROPE, if `optim_config.learn_pi` is `False`, and DROPL, if `True`. Data and configuration types are defined in `drorl/types.py`. 
* `drorl/helpers.py` contains helpers for regression, e.g. `PropensityPredictor` and `RegressorPredictor`, both of which use LightGBM. This file also contains a helper for continuum-of-outcomes regression, e.g. `ContinuumRegressorPredictor`, which uses `sklearn.ensemble.RandomForestRegressor`, to learn weights that approximate the conditional distribution of rewards given state and action. These weights can then be used to calculate the continuum nuisance, which is explained in our paper.
* `drorl/trainer.py` defines a Softmax policy and contains a neural network trainer for policy learning.

## Quick Start
Create an environment and install requirements.
```
conda create -n dro python=3.9
conda activate dro
pip install -r requirements.txt
```
If you have access to a GPU, install the GPU version of Jax. We recommend running DROPL experiments on a GPU.
```
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_releases.html
python -c "from jax.lib import xla_bridge; print(xla_bridge.get_backend().platform)"
```
If installed properly, this should print `gpu`. 

To generate datasets necessary for later steps, run
```
python generate_dataset.py
```
You should now see a `data/` folder containing all the datasets by seeds. \
To run DROPE experiments,
```
python eval_exps.py
```
To run DROPL experiments,
```
python learning_exps.py
```
Results are stored in pickle files. \
Finally, to visualize results, check out the `Visualize.ipynb` Jupyter notebook.

Thanks for checking out our work! Please don't hesitate to reach out with questions, comments, or suggestions! 

## Paper and Citation
ArXiv: https://arxiv.org/abs/2202.09667 \
ICML 2022 Proceedings: TBD \
Citation/BibTeX: TBD
