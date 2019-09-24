import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from abc import ABC, abstractmethod
import math

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
std_to_rho = lambda x: math.log(math.exp(x) - 1)


def kl_between_gaussians(p_mean, p_var, q_mean, q_var):
    """KL(p||q)"""
    return 0.5 * ((q_var / p_var).log() + (p_var + (p_mean - q_mean).pow(2)) / q_var - 1)


class BayesianLayer(ABC):

    @abstractmethod
    def layer_kl(self):
        pass


class BayesianLinear(nn.Module, BayesianLayer):
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
        super(BayesianLinear, self).__init__()
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
        W_kl = kl_between_gaussians(
            self.W_mean, F.softplus(self.W_rho).pow(2), self.W_prior_mean, self.W_prior_var
        )
        b_kl = kl_between_gaussians(
            self.b_mean, F.softplus(self.b_rho).pow(2), self.b_prior_mean, self.b_prior_var
        )
        return W_kl.sum() + b_kl.sum()


if __name__ == '__main__':
    bayesian_linear = BayesianLinear(3, 5, 'relu', 0.003, 'global')
    def test_kl():
        mean = torch.Tensor([3.0])
        var = torch.Tensor([5.0])
        zero = kl_between_gaussians(mean, var, mean, var)
        assert zero.isclose(torch.Tensor([0]))

        p_mean = torch.ones([500, 500]) * 2
        p_var = torch.ones([500, 500]) * 3
        q_mean = torch.ones([500, 500]) * 5
        q_var = torch.ones([500, 500]) * 4
        kl = kl_between_gaussians(
            p_mean, p_var, q_mean, q_var
        )
        # kl2 = torch.log(q_var.sqrt() / p_var.sqrt()) + (p_var + (p_mean - q_mean).pow(2)) / (2 * q_var) - 0.5
        # assert torch.isclose(kl, kl2)
        print(kl.shape)
    test_kl()
