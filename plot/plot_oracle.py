import sys

sys.path.insert(1, '../')
from src.model.base.lenet import LeNet
from src.model.bayesian.bayesian_lenet import BayesianLeNet
from src.model.bayesian.bayesian_classifier import BayesianClassifier
from copy import deepcopy
import torch
import torch.nn.functional as F
import ipdb
import matplotlib.pyplot as plt

dummy_net = LeNet(input_shape=[1, 28, 28], n_output=10)

bayesian_lenet = BayesianLeNet(
    lenet_prior_mean=dummy_net,
    lenet_posterior_mean=deepcopy(dummy_net),
    prior_stddev=0.001,
    optimize_prior_mean=False,
    optimize_prior_rho=False,
    optimize_posterior_mean=False,
    optimize_posterior_rho=True
)

bayesian_classifier = BayesianClassifier(
    bayesian_net=bayesian_lenet,
    prob_threshold=1e-4,
    oracle_prior_variance=True
)

bayesian_classifier.load_state_dict(torch.load('./data/wandb/run-20191007_004434-cmtk7zi2/model.pt'))

list_of_dict_of_params = [l.extract_parameters() for l in bayesian_classifier.bayesian_layers]
prior_mean = torch.cat([d['prior_mean'] for d in list_of_dict_of_params])
posterior_mean = torch.cat([d['posterior_mean'] for d in list_of_dict_of_params])
posterior_rho = torch.cat([d['posterior_rho'] for d in list_of_dict_of_params])
posterior_var = F.softplus(posterior_rho).pow(2).detach().numpy()
squared_l2 = (posterior_mean - prior_mean).pow(2).detach().numpy()

plt.scatter(posterior_var, squared_l2, alpha=0.3, s=3)
plt.xlabel('posterior_variance')
plt.ylabel('squared l2')
plt.xlim(posterior_var.min(), posterior_var.max())
plt.ylim(0, squared_l2.max())
plt.gca().ticklabel_format(style='sci', scilimits=(0, 0))
plt.savefig('./output/fashion_mnist_lenet_sgd_oracle_model.png')
