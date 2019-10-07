import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import ipdb
from abc import ABC, abstractmethod
from copy import deepcopy

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def inverse_softplus(x):
    return math.log(math.exp(x) - 1)


def kl_between_gaussians(p_mean, p_var, q_mean, q_var):
    """KL(p||q)"""
    return 0.5 * ((q_var / p_var).log() + (p_var + (p_mean - q_mean).pow(2)).div(q_var) - 1)


def kl_between_gaussians_oracle_prior_variance(posterior_mean, posterior_rho, prior_mean):
    posterior_var = F.softplus(posterior_rho).pow(2)
    return 0.5 * (1 + (posterior_mean - prior_mean).pow(2).div(posterior_var)).log()


def common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)


def var_of_rho(rho):
    return F.softplus(rho).pow(2)


def rho_of_var(var):
    return var.sqrt().exp().sub(1).log()


class BayesianLayer(nn.Module):
    def __init__(
            self,
            prior_mean,
            posterior_mean,
            prior_stddev,
            optimize_prior_mean,
            optimize_prior_rho,
            optimize_posterior_mean,
            optimize_posterior_rho
    ):
        super().__init__()
        assert (str(prior_mean) == str(posterior_mean))  # hacky, incomplete
        self.prior_mean = prior_mean
        self.posterior_mean = posterior_mean
        self.prior_stddev = prior_stddev

        self.prior_rho = deepcopy(prior_mean)
        self.posterior_rho = deepcopy(posterior_mean)

        with torch.no_grad():
            # initialize rho
            for rho_layer in [self.prior_rho, self.posterior_rho]:
                for p in rho_layer.parameters():
                    nn.init.constant_(p, inverse_softplus(self.prior_stddev))

            # # TODO: remove after debugging
            # for posterior_rho_p, prior_mean_p, posterior_mean_p in zip(self.posterior_rho.parameters(),
            #                                                            self.prior_mean.parameters(),
            #                                                            self.posterior_mean.parameters()):
            #     posterior_rho_p.copy_(rho_of_var(0.5 * (prior_mean_p - posterior_mean_p).pow(2)))

            # set requires_grad appropriately
            for (optimize, layer) in zip([optimize_prior_mean,
                                          optimize_prior_rho,
                                          optimize_posterior_mean,
                                          optimize_posterior_rho],
                                         [self.prior_mean,
                                          self.prior_rho,
                                          self.posterior_mean,
                                          self.posterior_rho]):
                if not optimize:
                    for p in layer.parameters():
                        p.requires_grad = False

    def perturb_posterior(self):
        mean_parameters = {name: p for (name, p) in self.posterior_mean.named_parameters()}
        rho_parameters = {name: p for (name, p) in self.posterior_rho.named_parameters()}

        noise_parameters = {name: torch.randn(p.shape, requires_grad=False).to(device)
                            for name, p in mean_parameters.items()}

        perturbed_parameters = {name: mean + F.softplus(rho) * noise
                                for name, mean, rho, noise in common_entries(mean_parameters,
                                                                             rho_parameters,
                                                                             noise_parameters)}
        return perturbed_parameters

    def extract_parameters(self):
        return {name: torch.cat([p.view(-1) for p in layer.parameters()])
                for (name, layer) in zip(['prior_mean', 'prior_rho', 'posterior_mean', 'posterior_rho'],
                                         [self.prior_mean, self.prior_rho, self.posterior_mean, self.posterior_rho])}

    def kl(self):
        parameter_vectors = self.extract_parameters()
        return kl_between_gaussians(
            p_mean=parameter_vectors['posterior_mean'],
            p_var=var_of_rho(parameter_vectors['posterior_rho']),
            q_mean=parameter_vectors['prior_mean'],
            q_var=var_of_rho(parameter_vectors['prior_rho'])
        ).sum()

    def kl_oracle_prior_variance(self):
        parameter_vectors = self.extract_parameters()
        return kl_between_gaussians_oracle_prior_variance(
            posterior_mean=parameter_vectors['posterior_mean'],
            posterior_rho=parameter_vectors['posterior_rho'],
            prior_mean=parameter_vectors['prior_mean']
        ).sum()


class BayesianConv2d(BayesianLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, mode):
        if mode == 'MC':
            perturbed_parameters = self.perturb_posterior()
            return F.conv2d(x, perturbed_parameters['weight'], perturbed_parameters.get('bias'),
                            self.posterior_mean.stride, self.posterior_mean.padding,
                            self.posterior_mean.dilation, self.posterior_mean.groups)
        elif mode == 'MAP':
            return self.posterior_mean(x)
        else:
            raise ValueError


class BayesianLinear(BayesianLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, x, mode):
        if mode == 'MC':
            perturbed_parameters = self.perturb_posterior()
            return F.linear(x, perturbed_parameters['weight'], perturbed_parameters.get('bias'))
        elif mode == 'MAP':
            return self.posterior_mean(x)
        else:
            raise ValueError


class BayesianBatchNorm2d(BayesianLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, input, mode):
        self.posterior_mean._check_input_dim(input)
        if mode == 'MC':
            # exponential_average_factor is self.posterior_mean.momentum set to
            # (when it is available) only so that if gets updated
            # in ONNX graph when this node is exported to ONNX.
            if self.posterior_mean.momentum is None:
                exponential_average_factor = 0.0
            else:
                exponential_average_factor = self.posterior_mean.momentum

            if self.posterior_mean.training and self.posterior_mean.track_running_stats:
                # TODO: if statement only here to tell the jit to skip emitting this when it is None
                if self.posterior_mean.num_batches_tracked is not None:
                    self.posterior_mean.num_batches_tracked += 1
                    if self.posterior_mean.momentum is None:  # use cumulative moving average
                        exponential_average_factor = 1.0 / float(self.posterior_mean.num_batches_tracked)
                    else:  # use exponential moving average
                        exponential_average_factor = self.posterior_mean.momentum

            perturbed_parameters = self.perturb_posterior()
            return F.batch_norm(
                input, self.posterior_mean.running_mean, self.posterior_mean.running_var,
                perturbed_parameters.get('weight'), perturbed_parameters.get('bias'),
                self.posterior_mean.training or not self.posterior_mean.track_running_stats,
                exponential_average_factor, self.posterior_mean.eps
            )
        elif mode == 'MAP':
            return self.posterior_mean(input)
        else:
            raise ValueError


if __name__ == '__main__':
    def test_bayesian_layers():
        prior_conv = nn.Conv2d(3, 5, 3, 1, 1, bias=False)
        posterior_conv = nn.Conv2d(3, 5, 3, 1, 1, bias=False)

        bayesian_conv = BayesianConv2d(prior_conv, posterior_conv, 0.01,
                                       False, False, True, True).to(device)

        assert bayesian_conv.prior_mean.weight.requires_grad is False

        prior_linear = nn.Linear(784, 200, bias=True)
        posterior_linear = nn.Linear(784, 200, bias=True)

        bayesian_linear = BayesianLinear(prior_linear, posterior_linear, 1e-5, False, False, True, True).to(device)
        x = torch.ones(3, 784).to(device)
        y1 = bayesian_linear(x, 'MC')
        y2 = bayesian_linear(x, 'MC')

        perturbed_parameters = bayesian_linear.perturb_posterior()


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

    # test_bayesian_layers()
    # test_kl()
