import torch
import torch.nn as nn
import torchvision
import ipdb
import numpy as np
import torch.utils.data as data
import math

from src.model import make_bnn_mlp
from src.model import BNN

from tqdm import tqdm

dataset_path = '/h/kylehsu/datasets'

torch.manual_seed(42)
np.random.seed(43)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameters
batch_size = 1280
n_samples = 1
n_epochs = 10
learning_rate = 1e-3

# Blundell: preprocess by dividing by 126. we divide by 255.

train_and_val_set = torchvision.datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_set_size = 50000

train_set, val_set = data.random_split(
    train_and_val_set, [train_set_size, len(train_and_val_set) - train_set_size])
train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
train_loader_eval = data.DataLoader(train_set, batch_size=len(train_set))
val_loader_eval = data.DataLoader(val_set, batch_size=len(val_set))

bnn = make_bnn_mlp(
    n_input=784,
    n_output=10,
    hidden_layer_sizes=[400, 400],
    W_prior_mean=0,
    W_prior_std=math.exp(-2),
    b_prior_mean=0,
    b_prior_std=math.exp(-2)
)
bnn = bnn.to(device)

# optim = torch.optim.SGD(
#     bnn.parameters(),
#     lr=learning_rate,
# )

optim = torch.optim.Adam(
    bnn.parameters(),
    lr=learning_rate
)


def evaluate(loader):
    bnn.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = bnn(x, 'MC')
            y_pred = out.argmax(dim=1)
            accuracy = (y == y_pred).sum().float().div(y.shape[0]).item()
        return accuracy


for i_epoch in tqdm(range(n_epochs)):
    bnn.train()
    kls = []
    log_likelihoods = []
    losses = []
    for x, y in tqdm(train_loader, total=len(train_set) // batch_size):
        x, y = x.to(device), y.to(device)

        kl, log_likelihood = bnn.forward_train(x, y, n_samples)
        loss = bnn.loss(kl, log_likelihood, train_set_size)
        optim.zero_grad()
        loss.backward()
        optim.step()

        kls.append(kl.item())
        log_likelihoods.append(log_likelihood.item())
        losses.append(loss.item())

    # eval
    acc_train = evaluate(train_loader_eval)
    acc_val = evaluate(val_loader_eval)

    print(f'epoch {i_epoch + 1}: acc_train {acc_train:.3f}, acc_val {acc_val:.3f}')
    print(f' kls: {np.mean(kls)}, \nlog_likelihoods: {np.mean(log_likelihoods)} '
          f'\nlosses: {np.mean(losses)}')




ipdb.set_trace()

x = 1