## Code accompanying "On the role of data in PAC-Bayes bounds", submitted to ICML 2020

A conda/pip environment spec is provided in `environment.yml`.

`data_dependent_prior_sgd.py` runs the SGD-characterization experiments whose results are plotted in Figure 4.

`data_dependent_prior_direct.py` runs the direct bound optimization experiments whose results are plotted in Figure 5.

`data_dependent_prior_ghost_sgd.py` runs the experiments whose results are plotted in the right column of Figure 7 (Appendix).

`data_dependent_prior_half_sgd.py` runs the experiments whose results are plotted in Figure 8 (Appendix).

`sweeps/` contains config files including all hyperparameters used for each run.