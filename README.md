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
To cite our work, please use the following citation. \
Kallus, N., Mao, X., Wang, K. &amp; Zhou, Z.. (2022). Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning. 
<i>Proceedings of the 39th International Conference on Machine Learning</i>, in <i>Proceedings of Machine Learning Research</i> 162:10598-10632 Available from https://proceedings.mlr.press/v162/kallus22a.html.

BibTex:
```
@InProceedings{pmlr-v162-kallus22a,
  title = 	 {Doubly Robust Distributionally Robust Off-Policy Evaluation and Learning},
  author =       {Kallus, Nathan and Mao, Xiaojie and Wang, Kaiwen and Zhou, Zhengyuan},
  booktitle = 	 {Proceedings of the 39th International Conference on Machine Learning},
  pages = 	 {10598--10632},
  year = 	 {2022},
  editor = 	 {Chaudhuri, Kamalika and Jegelka, Stefanie and Song, Le and Szepesvari, Csaba and Niu, Gang and Sabato, Sivan},
  volume = 	 {162},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {17--23 Jul},
  publisher =    {PMLR},
  pdf = 	 {https://proceedings.mlr.press/v162/kallus22a/kallus22a.pdf},
  url = 	 {https://proceedings.mlr.press/v162/kallus22a.html},
  abstract = 	 {Off-policy evaluation and learning (OPE/L) use offline observational data to make better decisions, which is crucial in applications where online experimentation is limited. However, depending entirely on logged data, OPE/L is sensitive to environment distribution shifts — discrepancies between the data-generating environment and that where policies are deployed. Si et al., (2020) proposed distributionally robust OPE/L (DROPE/L) to address this, but the proposal relies on inverse-propensity weighting, whose estimation error and regret will deteriorate if propensities are nonparametrically estimated and whose variance is suboptimal even if not. For standard, non-robust, OPE/L, this is solved by doubly robust (DR) methods, but they do not naturally extend to the more complex DROPE/L, which involves a worst-case expectation. In this paper, we propose the first DR algorithms for DROPE/L with KL-divergence uncertainty sets. For evaluation, we propose Localized Doubly Robust DROPE (LDR$^2$OPE) and show that it achieves semiparametric efficiency under weak product rates conditions. Thanks to a localization technique, LDR$^2$OPE only requires fitting a small number of regressions, just like DR methods for standard OPE. For learning, we propose Continuum Doubly Robust DROPL (CDR$^2$OPL) and show that, under a product rate condition involving a continuum of regressions, it enjoys a fast regret rate of $O(N^{-1/2})$ even when unknown propensities are nonparametrically estimated. We empirically validate our algorithms in simulations and further extend our results to general $f$-divergence uncertainty sets.}
}
```
