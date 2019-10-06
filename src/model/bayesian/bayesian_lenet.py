import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.bayesian.bayesian_layers import BayesianLinear, BayesianConv2d


class BayesianLeNet(nn.Module):

    def __init__(self, lenet_prior_mean, lenet_posterior_mean, **bayesian_kwargs):
        super().__init__()

        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0
        )
        self.conv_layers = nn.ModuleList()
        for conv_layer_prior_mean, conv_layer_posterior_mean in zip(lenet_prior_mean.conv_layers,
                                                                    lenet_posterior_mean.conv_layers):
            self.conv_layers.append(
                BayesianConv2d(
                    prior_mean=conv_layer_prior_mean,
                    posterior_mean=conv_layer_posterior_mean,
                    **bayesian_kwargs
                )
            )

        self.linear_layers = nn.ModuleList()
        for linear_layer_prior_mean, linear_layer_posterior_mean in zip(lenet_prior_mean.linear_layers,
                                                                        lenet_posterior_mean.linear_layers):
            self.linear_layers.append(
                BayesianLinear(
                    prior_mean=linear_layer_prior_mean,
                    posterior_mean=linear_layer_posterior_mean,
                    **bayesian_kwargs
                )
            )

    def forward(self, x, mode):
        assert x.dim() == 4
        for conv_layer in self.conv_layers:
            x = self.pool(conv_layer(x, mode))
        x = x.view(x.shape[0], -1)
        for i, linear_layer in enumerate(self.linear_layers):
            x = linear_layer(x, mode)
            if i < len(self.linear_layers) - 1:
                x = F.relu(x)
        assert x.dim() == 2
        x = F.softmax(x, dim=-1)
        return x
