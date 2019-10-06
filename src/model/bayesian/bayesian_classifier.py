import torch
import torch.nn as nn
import math

from src.model.base.classifier import Classifier
from src.model.bayesian.bayesian_layers import BayesianLayer

neg_half_log_2_pi = -0.5 * math.log(2.0 * math.pi)
softplus = lambda x: math.log(1 + math.exp(x))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class BayesianClassifier(nn.Module):
    def __init__(
            self,
            bayesian_net,
            prob_threshold,
            normalize_surrogate_by_log_classes=True,
            oracle_prior_variance=False
    ):
        super().__init__()
        self.net = bayesian_net
        self.prob_threshold = prob_threshold
        self.normalize_surrogate_by_log_classes = normalize_surrogate_by_log_classes
        self.bayesian_layers = self._find_bayesian_layers()
        self.oracle_prior_variance = oracle_prior_variance

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
            if self.oracle_prior_variance:
                kl += layer.kl_oracle_prior_variance()
            else:
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

    @staticmethod
    def surrogate(probs, y, prob_threshold, normalize_surrogate_by_log_classes):
        y = y.view([y.shape[0], -1])
        log_likelihood = probs.gather(1, y).clamp(min=prob_threshold, max=1).log()
        surrogate = -log_likelihood
        if normalize_surrogate_by_log_classes:
            n_classes = probs.shape[-1]
            surrogate = surrogate.div(math.log(n_classes))
        return surrogate

    @staticmethod
    def quad_bound(risk, kl, dataset_size, delta):
        log_2_sqrt_n_over_delta = math.log(2 * math.sqrt(dataset_size) / delta)
        fraction = (kl + log_2_sqrt_n_over_delta).div(2 * dataset_size)
        sqrt1 = (risk + fraction).sqrt()
        sqrt2 = fraction.sqrt()
        return (sqrt1 + sqrt2).pow(2)

    @staticmethod
    def pinsker_bound(risk, kl, dataset_size, delta):
        B = (kl + math.log(2 * math.sqrt(dataset_size) / delta)).div(dataset_size)
        return risk + B.div(2).sqrt()

    @staticmethod
    def inverted_kl_bound(risk, kl, dataset_size, delta):
        return torch.min(
            BayesianClassifier.quad_bound(risk, kl, dataset_size, delta),
            BayesianClassifier.pinsker_bound(risk, kl, dataset_size, delta)
        )

