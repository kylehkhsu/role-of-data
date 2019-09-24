import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, net):
        super(Classifier, self).__init__()
        self.add_module('net', net)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def loss(probs, y):
        y = y.view([y.shape[0], -1])
        log_likelihood = probs.gather(1, y).log().mean()
        return -log_likelihood

    @staticmethod
    def evaluate(probs, y):
        assert y.dim() == 1
        predictions = probs.argmax(dim=-1)
        correct = (predictions == y).sum().float()
        total = torch.tensor(y.shape[0])
        return correct, total
