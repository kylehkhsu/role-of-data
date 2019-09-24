import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb

class MLP(nn.Module):
    def __init__(self, n_input, n_output, hidden_layer_sizes):
        super(MLP, self).__init__()
        n_in = n_input
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            self.hidden_layers.append(nn.Linear(n_in, hidden_layer_sizes[i], bias=True))
            n_in = hidden_layer_sizes[i]
        self.output_layer = nn.Linear(n_in, n_output, bias=True)

    def forward(self, x):
        x = x.view([x.shape[0], -1])
        for hidden_layer in self.hidden_layers:
            x = hidden_layer(x)
            x = F.relu(x)
        x = self.output_layer(x)
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


if __name__ == '__main__':
    mlp = MLP(784, 10, [600])
    p = list(mlp.parameters())
    x = torch.ones(3, 28, 28)
    y = mlp(x)
    ipdb.set_trace()