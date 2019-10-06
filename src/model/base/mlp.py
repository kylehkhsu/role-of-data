import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb
import math


class MLP(nn.Module):
    def __init__(self, n_input, n_output, hidden_layer_sizes):
        super(MLP, self).__init__()
        n_in = n_input
        self.hidden_layers = nn.ModuleList()
        for i in range(len(hidden_layer_sizes)):
            layer = nn.Linear(n_in, hidden_layer_sizes[i], bias=True)
            init_std = 1 / math.sqrt(n_input)
            with torch.no_grad():
                nn.init.normal_(layer.weight, mean=0, std=0.1).clamp_(min=-0.2, max=0.2)
                nn.init.normal_(layer.bias, mean=0, std=0.1).clamp_(min=-0.2, max=0.2)
                # nn.init.normal_(layer.weight, mean=0, std=init_std).clamp_(min=-2 * init_std, max=2 * init_std)
                # nn.init.zeros_(layer.bias)
            self.hidden_layers.append(layer)
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


if __name__ == '__main__':
    mlp = MLP(784, 10, [600])
    p = list(mlp.parameters())
    x = torch.ones(3, 28, 28)
    y = mlp(x)
    ipdb.set_trace()
