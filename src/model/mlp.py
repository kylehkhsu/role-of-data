import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, n_input, n_output, hidden_layer_sizes):
        super(MLP, self).__init__()
        assert len(hidden_layer_sizes) == 2
        self.fc1 = nn.Linear(n_input, hidden_layer_sizes[0], bias=True)
        self.fc2 = nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1], bias=True)
        self.fc3 = nn.Linear(hidden_layer_sizes[1], n_output, bias=True)

    def forward(self, x):
        x = x.view([x.shape[0], -1])
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.softmax(x, dim=-1)
        return x

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
