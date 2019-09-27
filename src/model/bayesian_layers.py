import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
import math
import ipdb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
inverse_softplus = lambda x: math.log(math.exp(x) - 1)


def kl_between_gaussians(p_mean, p_var, q_mean, q_var):
    """KL(p||q)"""
    return 0.5 * ((q_var / p_var).log() + (p_var + (p_mean - q_mean).pow(2)) / q_var - 1)


def perturb_parameter(parameter_mean, parameter_rho):
    parameter_noise = torch.randn(parameter_mean.shape, requires_grad=False).to(device)
    parameter_perturbed = parameter_mean + F.softplus(parameter_rho) * parameter_noise
    return parameter_perturbed


def layer_kl(w_posterior_mean, w_posterior_rho, w_prior_mean, w_prior_rho,
             b_posterior_mean, b_posterior_rho, b_prior_mean, b_prior_rho):
    """KL[q(w|\theta) || p(w)]"""
    w_kl = kl_between_gaussians(
        w_posterior_mean,
        F.softplus(w_posterior_rho).pow(2),
        w_prior_mean,
        F.softplus(w_prior_rho).pow(2)
    )
    b_kl = kl_between_gaussians(
        b_posterior_mean,
        F.softplus(b_posterior_rho).pow(2),
        b_prior_mean,
        F.softplus(b_prior_rho).pow(2)
    )
    return w_kl.sum() + b_kl.sum()


def finish_init(
        bl: nn.Module,  # bayesian_layer
        w_prior_mean_init,
        b_prior_mean_init,
        w_posterior_mean_init,
        b_posterior_mean_init,
        w_mean_init_fn_,
        b_mean_init_fn_
):
    with torch.no_grad():  # we're modifying leaf variables with requires_grad=True in-place
        # initialize prior mean
        if w_prior_mean_init is None:
            w_mean_init_fn_(bl.w_prior_mean)
        else:
            bl.w_prior_mean = bl.w_prior_mean.copy_(w_prior_mean_init)

        if b_prior_mean_init is None:
            b_mean_init_fn_(bl.b_prior_mean)
        else:
            bl.b_prior_mean = bl.b_prior_mean.copy_(b_prior_mean_init)

        # construct posterior mean
        bl.w_posterior_mean = nn.Parameter(torch.empty_like(bl.w_prior_mean))
        bl.b_posterior_mean = nn.Parameter(torch.empty_like(bl.b_prior_mean))

        # initialize posterior mean
        if w_posterior_mean_init is None:
            bl.w_posterior_mean = bl.w_posterior_mean.copy_(bl.w_prior_mean)
        else:
            bl.w_posterior_mean = bl.w_posterior_mean.copy_(w_posterior_mean_init)

        if b_posterior_mean_init is None:
            bl.b_posterior_mean = bl.b_posterior_mean.copy_(bl.b_prior_mean)
        else:
            bl.b_posterior_mean = bl.b_posterior_mean.copy_(b_posterior_mean_init)

        # construct prior rho
        bl.w_prior_rho = nn.Parameter(torch.empty_like(bl.w_prior_mean))
        bl.b_prior_rho = nn.Parameter(torch.empty_like(bl.b_prior_mean))

        # initialize prior rho
        nn.init.constant_(bl.w_prior_rho, inverse_softplus(bl.prior_stddev))
        nn.init.constant_(bl.b_prior_rho, inverse_softplus(bl.prior_stddev))

        # construct posterior rho
        bl.w_posterior_rho = nn.Parameter(torch.empty_like(bl.w_prior_mean))
        bl.b_posterior_rho = nn.Parameter(torch.empty_like(bl.b_prior_mean))

        # initialize posterior rho
        nn.init.constant_(bl.w_posterior_rho, inverse_softplus(bl.prior_stddev))
        nn.init.constant_(bl.b_posterior_rho, inverse_softplus(bl.prior_stddev))

        if not bl.optimize_prior_mean:
            bl.w_prior_mean.requires_grad = False
            bl.b_prior_mean.requires_grad = False
        if not bl.optimize_prior_rho:
            bl.w_prior_rho.requires_grad = False
            bl.b_prior_rho.requires_grad = False
        if not bl.optimize_posterior_mean:
            bl.w_posterior_mean.requires_grad = False
            bl.b_posterior_mean.requires_grad = False
        if not bl.optimize_posterior_rho:
            bl.w_posterior_rho.requires_grad = False
            bl.b_posterior_rho.requires_grad = False


class BayesianLinear(nn.Module):
    def __init__(
            self,
            in_features,
            out_features,
            activation,
            prior_stddev,
            optimize_prior_mean,
            optimize_prior_rho,
            optimize_posterior_mean,
            optimize_posterior_rho,
            w_prior_mean_init=None,
            b_prior_mean_init=None,
            w_posterior_mean_init=None,
            b_posterior_mean_init=None,
    ):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_stddev = prior_stddev
        self.optimize_prior_mean = optimize_prior_mean
        self.optimize_prior_rho = optimize_prior_rho
        self.optimize_posterior_mean = optimize_posterior_mean
        self.optimize_posterior_rho = optimize_posterior_rho

        self.w_prior_mean = nn.Parameter(torch.Tensor(out_features, in_features))
        self.b_prior_mean = nn.Parameter(torch.Tensor(out_features))

        def w_mean_init_fn_(w_mean):
            std = 1 / math.sqrt(in_features)
            nn.init.normal_(w_mean, mean=0., std=std).clamp_(min=-2*std, max=2*std)

        # nn.init.normal_(bl.w_prior_mean, mean=0, std=bl.prior_stddev).clamp_(min=-2*bl.prior_stddev, max=2*bl.prior_stddev)
        # nn.init.normal_(bl.w_prior_mean, mean=0, std=0.1).clamp_(min=-0.2, max=0.2)
        # nn.init.normal_(bl.w_prior_mean, mean=0, std=mean_init_std).clamp_(min=-2 * mean_init_std,
        #                                                                    max=2 * mean_init_std)

        # nn.init.normal_(bl.b_prior_mean, mean=0, std=bl.prior_stddev).clamp_(min=-2*bl.prior_stddev, max=2*bl.prior_stddev)
        # nn.init.normal_(bl.b_prior_mean, mean=0, std=0.1).clamp_(min=-0.2, max=0.2)
        b_mean_init_fn_ = nn.init.zeros_

        finish_init(self,
                    w_prior_mean_init=w_prior_mean_init,
                    b_prior_mean_init=b_prior_mean_init,
                    w_posterior_mean_init=w_posterior_mean_init,
                    b_posterior_mean_init=b_posterior_mean_init,
                    w_mean_init_fn_=w_mean_init_fn_,
                    b_mean_init_fn_=b_mean_init_fn_
                    )
        self.activation = None
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softmax':
            self.activation = partial(F.softmax, dim=-1)
        else:
            raise ValueError

    def forward(self, x, mode):
        assert mode in ['MAP', 'MC']
        x = x.view([x.shape[0], -1])    # from N,C,H,W to N,C*H*W

        if mode == 'MAP':
            return self.activation(F.linear(x, self.w_posterior_mean, self.b_posterior_mean))

        w = perturb_parameter(self.w_posterior_mean, self.w_posterior_rho)
        b = perturb_parameter(self.b_posterior_mean, self.b_posterior_rho)
        z = self.activation(F.linear(x, w, b))

        if mode == 'MC':
            return z

    def kl(self):
        return layer_kl(
            self.w_posterior_mean, self.w_posterior_rho, self.w_prior_mean, self.w_prior_rho,
            self.b_posterior_mean, self.b_posterior_rho, self.b_prior_mean, self.b_prior_rho
        )


class BayesianConv2d(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size: tuple,
            activation,
            prior_stddev,
            optimize_prior_mean,
            optimize_prior_rho,
            optimize_posterior_mean,
            optimize_posterior_rho,
            w_prior_mean_init=None,
            b_prior_mean_init=None,
            w_posterior_mean_init=None,
            b_posterior_mean_init=None,
            stride=(1, 1),
            padding=(0, 0),
            dilation=(1, 1),
            groups=1,
    ):
        super(BayesianConv2d, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.prior_stddev = prior_stddev
        self.optimize_prior_mean = optimize_prior_mean
        self.optimize_prior_rho = optimize_prior_rho
        self.optimize_posterior_mean = optimize_posterior_mean
        self.optimize_posterior_rho = optimize_posterior_rho

        self.w_prior_mean = nn.Parameter(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size
        ))
        self.b_prior_mean = nn.Parameter(torch.Tensor(out_channels))

        def w_mean_init_fn_(w_mean):
            nn.init.kaiming_uniform_(w_mean, a=math.sqrt(5))

        def b_mean_init_fn_(b_mean):
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.w_prior_mean)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(b_mean, -bound, bound)

        finish_init(self,
                    w_prior_mean_init=w_prior_mean_init,
                    b_prior_mean_init=b_prior_mean_init,
                    w_posterior_mean_init=w_posterior_mean_init,
                    b_posterior_mean_init=b_posterior_mean_init,
                    w_mean_init_fn_=w_mean_init_fn_,
                    b_mean_init_fn_=b_mean_init_fn_
                    )
        self.activation = None
        if activation == 'relu':
            self.activation = F.relu
        elif activation == 'softmax':
            self.activation = partial(F.softmax, dim=-1)
        elif activation == 'maxpool':
            self.activation = partial(F.max_pool2d, kernel_size=2, stride=2, padding=0)
        else:
            raise ValueError

    def forward(self, x, mode):
        assert mode in ['MAP', 'MC']

        if mode == 'MAP':
            return self.activation(F.conv2d(x, self.w_posterior_mean, self.b_posterior_mean,
                                            self.stride, self.padding, self.dilation, self.groups))

        w = perturb_parameter(self.w_posterior_mean, self.w_posterior_rho)
        b = perturb_parameter(self.b_posterior_mean, self.b_posterior_rho)
        z = self.activation(F.conv2d(x, w, b,
                                     self.stride, self.padding, self.dilation, self.groups))

        if mode == 'MC':
            return z

    def kl(self):
        return layer_kl(
            self.w_posterior_mean, self.w_posterior_rho, self.w_prior_mean, self.w_prior_rho,
            self.b_posterior_mean, self.b_posterior_rho, self.b_prior_mean, self.b_prior_rho
        )

if __name__ == '__main__':
    def test_bayesian_linear():
        bayesian_linear = BayesianLinear(
            in_features=784,
            out_features=600,
            activation='relu',
            prior_stddev=0.003,
            optimize_prior_mean=False,
            optimize_prior_rho=False,
            optimize_posterior_mean=True,
            optimize_posterior_rho=True,
            w_prior_mean_init=None,
            b_prior_mean_init=None,
            w_posterior_mean_init=None,
            b_posterior_mean_init=None,
        )
        bayesian_linear = bayesian_linear.to(device)
        x = torch.ones(3, 784).to(device)
        y, kl = bayesian_linear(x, 'forward')
        assert kl == 0

        summary = [(name, p.shape, p.requires_grad) for name, p in bayesian_linear.named_parameters()]
        print(summary)

    def test_bayesian_conv2d():
        bayesian_conv2d = BayesianConv2d(
            in_channels=1,
            out_channels=20,
            kernel_size=(5, 5),
            activation='maxpool',
            prior_stddev=0.003,
            optimize_prior_mean=False,
            optimize_prior_rho=False,
            optimize_posterior_mean=True,
            optimize_posterior_rho=True,
        )
        bayesian_conv2d = bayesian_conv2d.to(device)
        x = torch.ones([3, 1, 28, 28]).to(device)
        y = bayesian_conv2d(x, 'MC')
        ipdb.set_trace()


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


    # test_bayesian_linear()
    test_bayesian_conv2d()
    # test_kl()
