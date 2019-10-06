import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from src.model.base.resnet import BasicBlock
from src.model.bayesian.bayesian_layers import BayesianLinear, BayesianConv2d, BayesianBatchNorm2d

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


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
