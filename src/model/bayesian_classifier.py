import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .bayesian_layers import BayesianLinear
from functools import partial
import numpy as np
import ipdb

neg_half_log_2_pi = -0.5 * math.log(2.0 * math.pi)
softplus = lambda x: math.log(1 + math.exp(x))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BayesianClassifier(nn.Module):
    def __init__(self, min_prob, *layers):
        super(BayesianClassifier, self).__init__()
        for i_layer, layer in enumerate(layers):
            self.add_module(f'layer{i_layer}', layer)

        self.layers = [module for name, module in self.named_modules() if 'layer' in name]
        self.min_prob = min_prob

    def forward(self, x, mode):
        x = x.view([x.shape[0], -1])

        if mode == 'forward':
            net_kl = 0.0
            for layer in self.layers:
                x, layer_kl = layer(x, mode)
                net_kl += layer_kl

            return x, net_kl
        else:
            for layer in self.layers:
                x = layer(x, mode)
            return x

    def forward_train(self, x, y, n_samples=1):
        x = x.view([x.shape[0], -1])
        y = y.view([y.shape[0], -1])

        running_kl = 0.0
        running_surrogate = 0.0
        running_correct = 0.0
        for i in range(n_samples):
            probs, kl = self(x, 'forward')
            n_classes = probs.shape[-1]

            log_likelihood = probs.gather(1, y).clamp(min=self.min_prob, max=1).log().mean()
            surrogate = log_likelihood.div(math.log(n_classes))
            predictions = probs.argmax(dim=-1)
            correct = (predictions == y.squeeze()).sum().float()

            running_correct += correct
            running_kl += kl
            running_surrogate += surrogate

        return running_kl / n_samples, running_surrogate / n_samples, running_correct / n_samples

    @staticmethod
    def quad_bound(risk, kl, dataset_size, delta):
        log_2_sqrt_n_over_delta = math.log(2 * math.sqrt(dataset_size) / delta)
        fraction = (kl + log_2_sqrt_n_over_delta).div(2 * dataset_size)
        sqrt1 = (risk + fraction).sqrt()
        sqrt2 = fraction.sqrt()
        return (sqrt1 + sqrt2).pow(2)

    @staticmethod
    def lambda_bound(risk, kl, dataset_size, delta, lam):
        log_2_sqrt_n_over_delta = math.log(2 * math.sqrt(dataset_size) / delta)
        term1 = risk.div(1 - lam / 2)
        term2 = (kl + log_2_sqrt_n_over_delta).div(dataset_size * lam * (1 - lam / 2))
        return term1 + term2

    @staticmethod
    def pinsker_bound(risk, kl, dataset_size, delta):
        B = (kl + math.log(2 * math.sqrt(dataset_size) / delta)).div(dataset_size)
        return risk + B.div(2).sqrt()

    @staticmethod
    def inverted_kl_bound(risk, kl, dataset_size, delta):
        return torch.min(
            BayesianClassifier.quad_bound(risk, kl, dataset_size, delta),
            BayesianClassifier.pinsker_bound(risk, kl, dataset_size, delta)
        )


def make_bmlp(n_input, n_output, hidden_layer_sizes, prior_std, min_prob, reparam_trick):
    layers = []
    n_in = n_input
    assert len(hidden_layer_sizes) > 0

    for h in hidden_layer_sizes:
        layers.append(
            BayesianLinear(
                n_in, h, 'relu', prior_std, reparam_trick
            )
        )
        n_in = h
    layers.append(
        BayesianLinear(
            n_in, n_output, 'softmax', prior_std, reparam_trick
        )
    )

    return BayesianClassifier(min_prob, *layers)


def make_bayesian_classifier_from_mlps(
        mlp_posterior_mean_init,
        mlp_prior_mean,
        prior_std,
        min_prob,
        reparam_trick,
        optimize_posterior_mean=True
):
    posterior_mean_init_parameters = list(mlp_posterior_mean_init.parameters())
    prior_mean_parameters = list(mlp_prior_mean.parameters())
    n_layers = len(prior_mean_parameters) // 2

    layers = []
    for i_layer in range(n_layers):
        if i_layer == n_layers - 1:
            activation = 'softmax'
        else:
            activation = 'relu'
        layers.append(
            BayesianLinear(
                n_input=prior_mean_parameters[i_layer * 2].shape[1],  # unintuitive, but correct
                n_output=prior_mean_parameters[i_layer * 2].shape[0],
                activation=activation,
                prior_std=prior_std,
                reparam_trick=reparam_trick,
                W_prior_mean=prior_mean_parameters[i_layer * 2].t(),
                b_prior_mean=prior_mean_parameters[i_layer * 2 + 1],
                W_posterior_mean_init=posterior_mean_init_parameters[i_layer * 2].t(),
                b_posterior_mean_init=posterior_mean_init_parameters[i_layer * 2 + 1],
                optimize_posterior_mean=optimize_posterior_mean
            )
        )
    return BayesianClassifier(min_prob, *layers)

