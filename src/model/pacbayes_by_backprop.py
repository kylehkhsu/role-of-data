import torch
import torch.nn as nn
import torch.nn.functional as F

import math
from functools import partial
import numpy as np
import ipdb

neg_half_log_2_pi = -0.5 * math.log(2.0 * math.pi)
softplus = lambda x: math.log(1 + math.exp(x))
std_to_rho = lambda x: math.log(math.exp(x) - 1)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BNNLinearLayer(nn.Module):
    def __init__(self, n_input, n_output, activation, prior_std):
        super(BNNLinearLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output

        # randomly initialize, then center prior distributions around the initialization
        W_prior_mean = torch.empty([self.n_input, self.n_output])
        nn.init.normal_(W_prior_mean, mean=0, std=0.1).clamp_(min=-0.2, max=0.2)

        b_prior_mean = torch.empty([self.n_output])
        nn.init.normal_(b_prior_mean, mean=0, std=0.1).clamp_(min=-0.2, max=0.2)

        self.W_mean = nn.Parameter(W_prior_mean)
        self.b_mean = nn.Parameter(b_prior_mean)

        self.register_buffer('W_prior_mean', W_prior_mean.clone().detach())
        self.register_buffer('b_prior_mean', b_prior_mean.clone().detach())

        self.W_rho = nn.Parameter(torch.ones([self.n_input, self.n_output]) * std_to_rho(prior_std))
        self.b_rho = nn.Parameter(torch.ones([self.n_output]) * std_to_rho(prior_std))

        self.register_buffer('W_prior_var', torch.Tensor([prior_std ** 2]))
        self.register_buffer('b_prior_var', torch.Tensor([prior_std ** 2]))

        self.activation = None
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softmax':
            self.activation = partial(F.softmax, dim=-1)
        else:
            raise ValueError

    def forward(self, x, mode):
        # uses the local reparameterization trick: https://arxiv.org/abs/1506.02557
        assert x.dim() == 2
        batch_size = x.shape[0]

        assert mode in ['forward', 'MAP', 'MC']

        z_mean = x.mm(self.W_mean) + self.b_mean

        if mode == 'MAP':
            return self.activation(z_mean)

        # sample from standard normal independently for each example
        z_noise = torch.randn([batch_size, self.n_output], requires_grad=False).to(device)
        z_std = (x.pow(2).mm(F.softplus(self.W_rho).pow(2)) + F.softplus(self.b_rho).pow(2)).sqrt()
        z = z_mean + z_std * z_noise
        z = self.activation(z)

        if mode == 'MC':
            return z

        # KL[q(w|\theta) || p(w)]
        W_kl = BNNLinearLayer.kl_between_gaussians(
            self.W_mean, F.softplus(self.W_rho).pow(2), self.W_prior_mean, self.W_prior_var
        )
        b_kl = BNNLinearLayer.kl_between_gaussians(
            self.b_mean, F.softplus(self.b_rho).pow(2), self.b_prior_mean, self.b_prior_var
        )
        layer_kl = W_kl.sum() + b_kl.sum()
        return z, layer_kl

    @staticmethod
    def kl_between_gaussians(p_mean, p_var, q_mean, q_var):
        """KL(p||q)"""
        return 0.5 * ((q_var / p_var).log() + (p_var + (p_mean - q_mean).pow(2)) / q_var - 1)


class BNN(nn.Module):
    def __init__(self, min_prob, *layers):
        super(BNN, self).__init__()

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

    def forward_train(self, x, y, n_samples):
        x = x.view([x.shape[0], -1])
        y = y.view([y.shape[0], -1])

        running_kl = 0.0
        running_log_likelihood = 0.0
        for i in range(n_samples):
            probs, kl = self(x, 'forward')

            log_likelihood = probs.gather(1, y).clamp(min=self.min_prob, max=1).log().mean()

            running_kl += kl
            running_log_likelihood += log_likelihood

        return running_kl / n_samples, running_log_likelihood / n_samples


def make_bnn_mlp(n_input, n_output, hidden_layer_sizes, prior_std, min_prob):
    layers = []
    n_in = n_input
    assert len(hidden_layer_sizes) > 0

    for h in hidden_layer_sizes:
        layers.append(
            BNNLinearLayer(
                n_in, h, 'relu', prior_std
            )
        )
        n_in = h
    layers.append(
        BNNLinearLayer(
            n_in, n_output, 'softmax', prior_std
        )
    )

    return BNN(min_prob, *layers)


if __name__ == '__main__':
    layer = BNNLinearLayer(784, 600, 'relu', 0.01)
    ipdb.set_trace()


    def test_kl():
        mean = torch.Tensor([3.0])
        var = torch.Tensor([5.0])
        zero = BNNLinearLayer.kl_between_gaussians(mean, var, mean, var)
        assert zero.isclose(torch.Tensor([0]))

        p_mean = torch.Tensor([3.0])
        p_var = torch.Tensor([5.0])
        q_mean = torch.Tensor([2.0])
        q_var = torch.Tensor([3.0])
        kl = BNNLinearLayer.kl_between_gaussians(
            p_mean, p_var, q_mean, q_var
        )
        kl2 = torch.log(q_var.sqrt() / p_var.sqrt()) + (p_var + (p_mean - q_mean).pow(2)) / (2 * q_var) - 0.5
        assert torch.isclose(kl, kl2)


    test_kl()