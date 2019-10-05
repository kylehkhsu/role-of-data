import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import ipdb
from src.model.bayesian_layers import inverse_softplus, kl_between_gaussians
from src.model.cnn import BasicBlock
from src.model.bayesian_classifier import BayesianClassifier
from src.model.classifier import Classifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def common_entries(*dcts):
    for i in set(dcts[0]).intersection(*dcts[1:]):
        yield (i,) + tuple(d[i] for d in dcts)


def var_of_rho(rho):
    return F.softplus(rho).pow(2)


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


class BayesianBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, basic_block_prior_mean, basic_block_posterior_mean, **bayesian_kwargs):
        super().__init__()
        self.conv1 = BayesianConv2d(
            prior_mean=basic_block_prior_mean.conv1,
            posterior_mean=basic_block_posterior_mean.conv1,
            **bayesian_kwargs
        )
        self.bn1 = BayesianBatchNorm2d(
            prior_mean=basic_block_prior_mean.bn1,
            posterior_mean=basic_block_posterior_mean.bn1,
            **bayesian_kwargs
        )
        self.conv2 = BayesianConv2d(
            prior_mean=basic_block_prior_mean.conv2,
            posterior_mean=basic_block_posterior_mean.conv2,
            **bayesian_kwargs
        )
        self.bn2 = BayesianBatchNorm2d(
            prior_mean=basic_block_prior_mean.bn2,
            posterior_mean=basic_block_posterior_mean.bn2,
            **bayesian_kwargs
        )

        self.shortcut = basic_block_posterior_mean.shortcut  # parameter-less

    def forward(self, x, mode):
        out = F.relu(self.bn1(self.conv1(x, mode), mode))
        out = self.bn2(self.conv2(out, mode), mode)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BayesianSequential(nn.Sequential):
    def forward(self, x, mode):
        for module in self._modules.values():
            x = module(x, mode)
        return x


class BayesianResNet(nn.Module):
    def __init__(self, resnet_prior_mean, resnet_posterior_mean, **bayesian_kwargs):
        super().__init__()
        self.conv1 = BayesianConv2d(
            prior_mean=resnet_prior_mean.conv1,
            posterior_mean=resnet_posterior_mean.conv1,
            **bayesian_kwargs
        )
        self.bn1 = BayesianBatchNorm2d(
            prior_mean=resnet_prior_mean.bn1,
            posterior_mean=resnet_posterior_mean.bn1,
            **bayesian_kwargs
        )
        self.layer1 = self._make_layer(resnet_prior_mean.layer1, resnet_posterior_mean.layer1, **bayesian_kwargs)
        self.layer2 = self._make_layer(resnet_prior_mean.layer2, resnet_posterior_mean.layer2, **bayesian_kwargs)
        self.layer3 = self._make_layer(resnet_prior_mean.layer3, resnet_posterior_mean.layer3, **bayesian_kwargs)
        self.linear = BayesianLinear(
            prior_mean=resnet_prior_mean.linear,
            posterior_mean=resnet_posterior_mean.linear,
            **bayesian_kwargs
        )

    def _make_layer(self, layer_prior_mean, layer_posterior_mean, **bayesian_kwargs):
        assert type(layer_prior_mean) == type(layer_posterior_mean) == nn.Sequential
        blocks = []
        for block_prior_mean, block_posterior_mean in zip(layer_prior_mean, layer_posterior_mean):
            if type(block_prior_mean) == type(block_posterior_mean) == BasicBlock:
                blocks.append(
                    BayesianBasicBlock(
                        basic_block_prior_mean=block_prior_mean,
                        basic_block_posterior_mean=block_posterior_mean,
                        **bayesian_kwargs
                    )
                )
            else:
                raise NotImplementedError
        return BayesianSequential(*blocks)

    def forward(self, x, mode):
        out = F.relu(self.bn1(self.conv1(x, mode), mode))
        out = self.layer1(out, mode)
        out = self.layer2(out, mode)
        out = self.layer3(out, mode)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out, mode)
        out = F.softmax(out, dim=-1)
        return out


class BayesianResNetClassifier(nn.Module):
    def __init__(
            self,
            bayesian_net,
            prob_threshold,
            normalize_surrogate_by_log_classes=True,
            **kwargs
    ):
        super().__init__()
        self.net = bayesian_net
        self.prob_threshold = prob_threshold
        self.normalize_surrogate_by_log_classes = normalize_surrogate_by_log_classes
        self.bayesian_layers = self._find_bayesian_layers()

    def forward(self, x, mode):
        return self.net(x, mode)

    def forward_train(self, x, y, n_samples=1):
        running_kl = 0.0
        running_surrogate = 0.0
        for i in range(n_samples):
            probs = self.forward(x, 'MC')
            kl = self.kl()
            surrogate = BayesianClassifier.surrogate(
                probs=probs,
                y=y,
                prob_threshold=self.prob_threshold,
                normalize_surrogate_by_log_classes=self.normalize_surrogate_by_log_classes
            ).mean()

            running_kl += kl
            running_surrogate += surrogate

        return running_kl / n_samples, running_surrogate / n_samples

    def evaluate_on_loader(self, loader):
        training = self.training
        self.eval()

        corrects, totals, surrogates = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                probs = self(x, 'MC')
                correct, total = Classifier.evaluate(probs, y)
                surrogate_batch = BayesianClassifier.surrogate(
                    probs=probs,
                    y=y,
                    prob_threshold=self.prob_threshold,
                    normalize_surrogate_by_log_classes=self.normalize_surrogate_by_log_classes
                )
                corrects += correct.item()
                totals += total.item()
                surrogates += surrogate_batch.sum().item()
        error = 1 - corrects / totals
        surrogate = surrogates / totals

        self.train(mode=training)
        return error, surrogate

    def kl(self):
        kl = 0.0
        for layer in self.bayesian_layers:
            kl += layer.kl()
        return kl

    def _find_bayesian_layers(self):
        bayesian_layers = []

        def __find_bayesian_layers(module):
            if isinstance(module, BayesianLayer):
                bayesian_layers.append(module)
            for m in module.children():
                __find_bayesian_layers(m)

        __find_bayesian_layers(self.net)
        return bayesian_layers


if __name__ == '__main__':
    def test_bayesian_layers():
        prior_conv = nn.Conv2d(3, 5, 3, 1, 1, bias=False)
        posterior_conv = nn.Conv2d(3, 5, 3, 1, 1, bias=False)

        bayesian_conv = BayesianConv2d(prior_conv, posterior_conv, 0.01,
                                       False, False, True, True)

        assert bayesian_conv.prior_mean.weight.requires_grad is False

        prior_linear = nn.Linear(784, 200, bias=True)
        posterior_linear = nn.Linear(784, 200, bias=True)

        bayesian_linear = BayesianLinear(prior_linear, posterior_linear, 0.01, False, False, True, True).to(device)
        x = torch.ones(3, 784).to(device)
        y1 = bayesian_linear(x)
        y2 = bayesian_linear(x)


    def test_bayesian_resnet():
        from src.model.cnn import resnet20

        resnet_prior_mean = resnet20().to(device)
        resnet_posterior_mean = resnet20().to(device)

        bayesian_resnet = BayesianResNet(
            resnet_prior_mean=resnet_prior_mean,
            resnet_posterior_mean=resnet_posterior_mean,
            prior_stddev=1e-10,
            optimize_prior_mean=False,
            optimize_prior_rho=False,
            optimize_posterior_mean=True,
            optimize_posterior_rho=True
        )
        bayesian_classifier = BayesianResNetClassifier(
            bayesian_net=bayesian_resnet,
            prob_threshold=1e-4,
            normalize_surrogate_by_log_classes=True
        ).to(device)

        x = torch.ones(5, 3, 32, 32).to(device)
        y = bayesian_classifier(x, 'MC')

        y1 = resnet_posterior_mean(x)
        y2 = bayesian_classifier(x, 'MAP')

        assert torch.allclose(y1, y2)

        assert len([l for l in bayesian_classifier.bayesian_layers if 'Conv2d' in str(l) or 'Linear' in str(l)]) == 20

        # kl == 0 if prior = posterior
        bayesian_resnet = BayesianResNet(
            resnet_prior_mean=resnet_posterior_mean,
            resnet_posterior_mean=resnet_posterior_mean,
            prior_stddev=1e-4,
            optimize_prior_mean=False,
            optimize_prior_rho=False,
            optimize_posterior_mean=True,
            optimize_posterior_rho=True
        )
        bayesian_classifier = BayesianResNetClassifier(
            bayesian_net=bayesian_resnet,
            prob_threshold=1e-4,
            normalize_surrogate_by_log_classes=True
        ).to(device)

        assert bayesian_classifier.kl() == 0



        ipdb.set_trace()


    test_bayesian_resnet()
