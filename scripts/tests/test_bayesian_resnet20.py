import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
from src.model.bayesian.bayesian_classifier import BayesianClassifier
from src.model.base.resnet import resnet20
from src.model.bayesian.bayesian_resnet import BayesianResNet

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test_bayesian_resnet():
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
    bayesian_classifier = BayesianClassifier(
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
    bayesian_classifier = BayesianClassifier(
        bayesian_net=bayesian_resnet,
        prob_threshold=1e-4,
        normalize_surrogate_by_log_classes=True,
        oracle_prior_variance=True
    ).to(device)

    assert bayesian_classifier.kl() == 0

    ipdb.set_trace()


test_bayesian_resnet()
