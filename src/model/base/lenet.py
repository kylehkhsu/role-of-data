import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    """https://github.com/anonymous-108794/nnet-compression/blob/master/nnet/models/lenet5.py"""

    def __init__(self, input_shape, n_output, *args, **kwargs):
        super(LeNet, self).__init__()
        assert len(input_shape) == 3
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0
        )
        self.conv_layers = nn.ModuleList()
        self.conv_layers.append(
            nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1, padding=0)
        )
        self.conv_layers.append(
            nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1, padding=0)
        )

        with torch.no_grad():
            x = torch.ones([1] + input_shape)
            for conv_layer in self.conv_layers:
                x = self.pool(conv_layer(x))
            x = x.view(x.shape[0], -1)
            n_in = x.shape[-1]
        self.linear_layers = nn.ModuleList()
        self.linear_layers.append(
            nn.Linear(n_in, 500, bias=True)
        )
        self.linear_layers.append(
            nn.Linear(500, 10, bias=True)
        )

    def forward(self, x):
        assert x.dim() == 4
        for conv_layer in self.conv_layers:
            x = self.pool(conv_layer(x))
        x = x.view(x.shape[0], -1)
        for i, linear_layer in enumerate(self.linear_layers):
            x = linear_layer(x)
            if i < len(self.linear_layers) - 1:
                x = F.relu(x)
        assert x.dim() == 2
        x = F.softmax(x, dim=-1)
        return x


if __name__ == '__main__':

    def test_lenet():
        lenet = LeNet([1, 28, 28], 10, [600])
        x = torch.ones(3, 1, 28, 28)
        lenet(x)