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
    def __init__(
            self,
            n_input,
            n_output,
            activation,
            prior_std,
            reparam_trick,
            W_prior_mean=None,
            b_prior_mean=None,
            W_posterior_mean_init=None,
            b_posterior_mean_init=None,
            optimize_posterior_mean=True,
            perturbed_posterior_variance_init=False,
    ):
        super(BNNLinearLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        assert reparam_trick in ['local', 'global']
        self.reparam_trick = reparam_trick

        if W_prior_mean is None and b_prior_mean is None:
            W_prior_mean = torch.empty([self.n_input, self.n_output])
            b_prior_mean = torch.empty([self.n_output])
            nn.init.normal_(W_prior_mean, mean=0, std=prior_std).clamp_(min=-2 * prior_std, max=2 * prior_std)
            nn.init.normal_(b_prior_mean, mean=0, std=prior_std).clamp_(min=-2 * prior_std, max=2 * prior_std)
        elif W_prior_mean is not None and b_prior_mean is not None:
            pass
        else:
            raise ValueError

        if W_posterior_mean_init is None and b_posterior_mean_init is None:
            self.W_mean = nn.Parameter(W_prior_mean)
            self.b_mean = nn.Parameter(b_prior_mean)
        elif W_posterior_mean_init is not None and b_posterior_mean_init is not None:
            if optimize_posterior_mean:
                self.W_mean = nn.Parameter(W_posterior_mean_init)
                self.b_mean = nn.Parameter(b_posterior_mean_init)
            else:
                self.register_buffer('W_mean', W_posterior_mean_init.clone().detach())
                self.register_buffer('b_mean', b_posterior_mean_init.clone().detach())
        else:
            raise ValueError

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
        assert mode in ['forward', 'MAP', 'MC']
        assert x.dim() == 2
        batch_size = x.shape[0]

        if self.reparam_trick == 'local':
            # uses the local reparameterization trick: https://arxiv.org/abs/1506.02557

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

            return z, self.layer_kl()
        elif self.reparam_trick == 'global':
            if mode == 'MAP':
                return self.activation(x.mm(self.W_mean) + self.b_mean)

            # sample from standard normal once for the entire batch
            W_noise = torch.randn(self.W_mean.shape, requires_grad=False).to(device)
            b_noise = torch.randn(self.b_mean.shape, requires_grad=False).to(device)
            W = self.W_mean + F.softplus(self.W_rho) * W_noise
            b = self.b_mean + F.softplus(self.b_rho) * b_noise
            z = self.activation(x.mm(W) + b)

            if mode == 'MC':
                return z

            return z, self.layer_kl()

    def layer_kl(self):
        """KL[q(w|\theta) || p(w)]"""
        W_kl = BNNLinearLayer.kl_between_gaussians(
            self.W_mean, F.softplus(self.W_rho).pow(2), self.W_prior_mean, self.W_prior_var
        )
        b_kl = BNNLinearLayer.kl_between_gaussians(
            self.b_mean, F.softplus(self.b_rho).pow(2), self.b_prior_mean, self.b_prior_var
        )
        return W_kl.sum() + b_kl.sum()

    @staticmethod
    def kl_between_gaussians(p_mean, p_var, q_mean, q_var):
        """KL(p||q)"""
        return 0.5 * ((q_var / p_var).log() + (p_var + (p_mean - q_mean).pow(2)) / q_var - 1)


class BMLP(nn.Module):
    def __init__(self, min_prob, *layers):
        super(BMLP, self).__init__()
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
            BMLP.quad_bound(risk, kl, dataset_size, delta),
            BMLP.pinsker_bound(risk, kl, dataset_size, delta)
        )


def make_bmlp(n_input, n_output, hidden_layer_sizes, prior_std, min_prob, reparam_trick):
    layers = []
    n_in = n_input
    assert len(hidden_layer_sizes) > 0

    for h in hidden_layer_sizes:
        layers.append(
            BNNLinearLayer(
                n_in, h, 'relu', prior_std, reparam_trick
            )
        )
        n_in = h
    layers.append(
        BNNLinearLayer(
            n_in, n_output, 'softmax', prior_std, reparam_trick
        )
    )

    return BMLP(min_prob, *layers)


def make_bmlp_from_mlps(mlp_posterior_mean_init, mlp_prior_mean, prior_std, min_prob, reparam_trick, optimize_posterior_mean=True):
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
            BNNLinearLayer(
                n_input=prior_mean_parameters[i_layer * 2].shape[1],    # unintuitive, but correct
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
    return BMLP(min_prob, *layers)


if __name__ == '__main__':
    # layer = BNNLinearLayer(784, 600, 'relu', 0.01)
    # ipdb.set_trace()


    def test_kl():
        mean = torch.Tensor([3.0])
        var = torch.Tensor([5.0])
        zero = BNNLinearLayer.kl_between_gaussians(mean, var, mean, var)
        assert zero.isclose(torch.Tensor([0]))

        p_mean = torch.ones([500, 500]) * 2
        p_var = torch.ones([500, 500]) * 3
        q_mean = torch.ones([500, 500]) * 5
        q_var = torch.ones([500, 500]) * 4
        kl = BNNLinearLayer.kl_between_gaussians(
            p_mean, p_var, q_mean, q_var
        )
        # kl2 = torch.log(q_var.sqrt() / p_var.sqrt()) + (p_var + (p_mean - q_mean).pow(2)) / (2 * q_var) - 0.5
        # assert torch.isclose(kl, kl2)
        print(kl.shape)

    test_kl()
