import torch
import torch.nn as nn

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class Classifier(nn.Module):
    def __init__(self, net):
        super(Classifier, self).__init__()
        self.add_module('net', net)

    def forward(self, x):
        return self.net(x)

    @staticmethod
    def cross_entropy(probs, y):
        """Returns cross entropy between probs and y, of the same shape of probs"""
        y = y.view([y.shape[0], -1])
        log_likelihood = probs.gather(1, y).log()
        return -log_likelihood

    @staticmethod
    def evaluate(probs, y):
        assert y.dim() == 1
        predictions = probs.argmax(dim=-1)
        correct = (predictions == y).sum().float()
        total = torch.tensor(y.shape[0])
        return correct, total

    def evaluate_on_loader(self, loader):
        training = self.training
        self.eval()

        corrects, totals, cross_entropies = 0.0, 0.0, 0.0
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(device), y.to(device)
                probs = self(x)
                correct, total = self.evaluate(probs, y)
                cross_entropy_batch = self.cross_entropy(probs, y)

                corrects += correct.item()
                totals += total.item()
                cross_entropies += cross_entropy_batch.sum().item()
        error = 1 - corrects / totals
        cross_entropy = cross_entropies / totals

        self.train(mode=training)

        return error, cross_entropy
