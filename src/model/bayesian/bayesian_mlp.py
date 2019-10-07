import torch
import torch.nn as nn
import torch.nn.functional as F
from src.model.bayesian.bayesian_layers import BayesianLinear


class BayesianMLP(nn.Module):
    def __init__(self, mlp_prior_mean, mlp_posterior_mean, **bayesian_kwargs):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        for hidden_layer_prior_mean, hidden_layer_posterior_mean in zip(mlp_prior_mean.hidden_layers,
                                                                        mlp_posterior_mean.hidden_layers):
            self.hidden_layers.append(
                BayesianLinear(
                    prior_mean=hidden_layer_prior_mean,
                    posterior_mean=hidden_layer_posterior_mean,
                    **bayesian_kwargs
                )
            )

        self.output_layer = BayesianLinear(
            prior_mean=mlp_prior_mean.output_layer,
            posterior_mean=mlp_posterior_mean.output_layer,
            **bayesian_kwargs
        )

    def forward(self, x, mode):
        x = x.view([x.shape[0], -1])
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x, mode)
            x = F.relu(x)
        x = self.output_layer(x, mode)
        x = F.softmax(x, dim=-1)
        return x
