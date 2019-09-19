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
    def __init__(self, n_input, n_output, activation, prior_std, reparam_trick, config):
        super(BNNLinearLayer, self).__init__()
        self.n_input = n_input
        self.n_output = n_output
        self.reparam_trick = reparam_trick

        # randomly initialize, then center prior distributions around the initialization
        W_prior_mean = torch.empty([self.n_input, self.n_output])
        # nn.init.normal_(W_prior_mean, mean=0, std=prior_std).clamp_(min=-2 * prior_std, max=2 * prior_std)
        nn.init.normal_(W_prior_mean, mean=0, std=0.1).clamp_(min=-0.2, max=0.2)    # waseem

        b_prior_mean = torch.empty([self.n_output])
        # nn.init.normal_(b_prior_mean, mean=0, std=prior_std).clamp_(min=-2 * prior_std, max=2 * prior_std)
        nn.init.normal_(b_prior_mean, mean=0, std=0.1).clamp_(min=-0.2, max=0.2)    # waseem

        self.W_mean = nn.Parameter(W_prior_mean)
        self.b_mean = nn.Parameter(b_prior_mean)

        self.register_buffer('W_prior_mean', W_prior_mean.clone().detach())
        self.register_buffer('b_prior_mean', b_prior_mean.clone().detach())

        if config.covariance_init_strategy == 'isotropic':
            self.W_rho = nn.Parameter(torch.ones([self.n_input, self.n_output]) * std_to_rho(prior_std))
            self.b_rho = nn.Parameter(torch.ones([self.n_output]) * std_to_rho(prior_std))

            self.register_buffer('W_prior_var', torch.Tensor([prior_std ** 2]))
            self.register_buffer('b_prior_var', torch.Tensor([prior_std ** 2]))
        elif config.covariance_init_strategy == 'diagonal':
            # https://github.com/deepmind/sonnet/blob/master/sonnet/examples/brnn_ptb.py#L276
            W_prior_rho = torch.empty([self.n_input, self.n_output])
            nn.init.uniform_(W_prior_rho, a=std_to_rho(prior_std / 2.0), b=std_to_rho(prior_std))

            b_prior_rho = torch.empty([self.n_output])
            nn.init.uniform_(b_prior_rho, a=std_to_rho(prior_std / 2.0), b=std_to_rho(prior_std))
            self.W_rho = nn.Parameter(W_prior_rho)
            self.b_rho = nn.Parameter(b_prior_rho)

            self.register_buffer('W_prior_var', F.softplus(W_prior_rho.clone().detach()).pow(2))
            self.register_buffer('b_prior_var', F.softplus(b_prior_rho.clone().detach()).pow(2))

        else:
            raise ValueError

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
        # KL[q(w|\theta) || p(w)]
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
        running_correct = 0.0
        for i in range(n_samples):
            probs, kl = self(x, 'forward')

            log_likelihood = probs.gather(1, y).clamp(min=self.min_prob, max=1).log().mean()
            predictions = probs.argmax(dim=-1)
            correct = (predictions == y.squeeze()).sum().float()

            running_correct += correct
            running_kl += kl
            running_log_likelihood += log_likelihood

        return running_kl / n_samples, running_log_likelihood / n_samples, running_correct / n_samples


def make_bnn_mlp(n_input, n_output, hidden_layer_sizes, prior_std, min_prob, reparam_trick, config):
    layers = []
    n_in = n_input
    assert len(hidden_layer_sizes) > 0

    for h in hidden_layer_sizes:
        layers.append(
            BNNLinearLayer(
                n_in, h, 'relu', prior_std, reparam_trick, config
            )
        )
        n_in = h
    layers.append(
        BNNLinearLayer(
            n_in, n_output, 'softmax', prior_std, reparam_trick, config
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
