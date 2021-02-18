## Code accompanying [On the role of data in PAC-Bayes bounds](https://arxiv.org/abs/2006.10929), AISTATS 2021

### Installation
```
git clone https://github.com/kylehkhsu/pacbayes-opt
cd pacbayes-opt
conda env create -f environment.yml
conda activate pacbayes-opt
```

### Login to Weights & Biases
1. Navigate to [https://wandb.ai/authorize](https://wandb.ai/authorize), login, and copy the key (`KEY`)
2. `wandb login KEY`


### Overview

`data_dependent_prior_sgd.py` runs the SGD-characterization experiments whose results are plotted in Figure 4.

`data_dependent_prior_direct.py` runs the direct bound optimization experiments whose results are plotted in Figure 5.

`data_dependent_prior_ghost_sgd.py` runs the experiments whose results are plotted in the right column of Figure 7 (Appendix).

`data_dependent_prior_half_sgd.py` runs the experiments whose results are plotted in Figure 8 (Appendix).

`sweeps/` contains config files including all hyperparameters used for the presented results.

### Running the code
You have two options:
1. Simply run a script, e.g. `python scripts/data_dependent_prior_sgd.py`
2. Use a config file in `sweeps/` to run a `wandb` sweep (see [this example](https://github.com/wandb/examples/tree/master/examples/keras/keras-cnn-fashion)).
As-is, each file specifies a sweep over 50 seeds.


### Citing this work
If you find the code and/or paper helpful for your research, please cite
```
@inproceedings{dziugaite2021role,
    title={On the role of data in PAC-Bayes bounds},
    author={Dziugaite, Gintare Karolina and Hsu, Kyle and Gharbieh, Waseem and Arpino, Gabriel and Roy, Daniel M},
    booktitle={International Conference on Artificial Intelligence and Statistics (AISTATS)},
    year={2021}
}
```
