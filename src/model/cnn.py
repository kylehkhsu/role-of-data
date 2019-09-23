import torch
import torch.nn as nn
import torch.nn.functional as F
import ipdb


class CNN(nn.Module):
    def __init__(self, n_input, n_output, hidden_layer_sizes):
        super(CNN, self).__init__()
        assert len(hidden_layer_sizes) == 5
        self.pad = nn.ZeroPad2d(padding=2)
        self.pool = nn.MaxPool2d(
            kernel_size=2, stride=2, padding=0
        )
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=hidden_layer_sizes[0], kernel_size=3, stride=1, padding=1
        )
        self.conv2 = nn.Conv2d(
            in_channels=hidden_layer_sizes[0], out_channels=hidden_layer_sizes[1], kernel_size=3, stride=1, padding=1
        )
        self.conv3 = nn.Conv2d(
            in_channels=hidden_layer_sizes[1], out_channels=hidden_layer_sizes[2], kernel_size=3, stride=1, padding=1
        )
        self.conv4 = nn.Conv2d(
            in_channels=hidden_layer_sizes[2], out_channels=hidden_layer_sizes[3], kernel_size=3, stride=1, padding=1
        )
        self.conv5 = nn.Conv2d(
            in_channels=hidden_layer_sizes[3], out_channels=hidden_layer_sizes[4], kernel_size=3, stride=1, padding=1
        )
        self.conv6 = nn.Conv2d(
            in_channels=hidden_layer_sizes[4], out_channels=n_output, kernel_size=3, stride=1, padding=1
        )
        # self.fc3 = nn.Linear(
        #     in_features=hidden_layer_sizes[1]*7*7, out_features=n_output
        # )

    def forward(self, x):
        assert x.dim() == 4
        x = self.pad(x)
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))
        x = self.pool(F.relu(self.conv5(x)))
        x = self.conv6(x)
        x = x.view(x.shape[0], -1)
        x = F.softmax(x, dim=-1)


        # x = self.conv1(x)
        # x = F.relu(x)
        # x = self.pool(x)
        # x = self.conv2(x)
        # x = F.relu(x)
        # x = self.pool(x)
        # # x = x.mean(dim=-3)  # should be the channel
        # x = x.view(x.shape[0], -1)
        # x = self.fc3(x)
        # x = F.softmax(x, dim=-1)
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
    cnn = CNN(784, 10, [64]*5)
    x = torch.ones(3, 1, 28, 28)
    cnn(x)
