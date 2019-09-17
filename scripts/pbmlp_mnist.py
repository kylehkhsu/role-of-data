import torch
import torch.nn as nn
import torchvision
import ipdb
import numpy as np
import torch.utils.data as data
import math

from src.model.pacbayes_by_backprop import make_bnn_mlp

from tqdm import tqdm

dataset_path = '/h/kylehsu/datasets'

torch.manual_seed(42)
np.random.seed(43)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# hyperparameters
batch_size = 256    # authors: 256
n_samples = 1
n_epochs = 100000 // (50000 // 256) * 2     # authors: 100K iterations, up to 1M
learning_rate = 0.01     # authors: 0.01 or 0.001
momentum = 0.95      # authors: 0.95 or 0.99
prior_std = 0.1     # authors: prior_var = 0.01
hidden_layer_sizes = [600, 600]     # authors: [600, 600]
min_prob = 1e-4     # authors: 1e-2, 1e-4, 1e-5

delta = 0.05    # karolina

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
    hidden_layer_sizes=hidden_layer_sizes,
    prior_std=prior_std,
    min_prob=min_prob
)
bnn = bnn.to(device)

optim = torch.optim.SGD(
    bnn.parameters(),
    lr=learning_rate,
    momentum=momentum
)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=1)


def evaluate(loader):
    bnn.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = bnn(x, 'MAP')
            y_pred = out.argmax(dim=1)
            accuracy = (y == y_pred).sum().float().div(y.shape[0]).item()
        return accuracy


def quad_bound(risk, kl, dataset_size, delta):
    log_2_sqrt_n_over_delta = math.log(2 * math.sqrt(dataset_size) / delta)
    fraction = (kl + log_2_sqrt_n_over_delta).div(2 * dataset_size)
    sqrt1 = (risk + fraction).sqrt()
    sqrt2 = fraction.sqrt()
    return (sqrt1 + sqrt2).pow(2)


def lambda_bound(risk, kl, dataset_size, delta, lam):
    raise NotImplementedError
    log_2_sqrt_n_over_delta = math.log(2 * math.sqrt(dataset_size) / delta)
    term1 = risk.div(1 - lam / 2)
    term2 = (kl + log_2_sqrt_n_over_delta).div(dataset_size * lam * (1 - lam / 2))
    return term1 + term2

bound = quad_bound

for i_epoch in tqdm(range(n_epochs)):
    bnn.train()
    kls = []
    log_likelihoods = []
    losses = []
    for x, y in tqdm(train_loader, total=len(train_set) // batch_size):
        x, y = x.to(device), y.to(device)

        kl, log_likelihood = bnn.forward_train(x, y, n_samples)
        loss = bound(-log_likelihood, kl, train_set_size, delta)
        optim.zero_grad()
        loss.backward()
        optim.step()

        kls.append(kl.item())
        log_likelihoods.append(log_likelihood.item())
        losses.append(loss.item())

    # eval
    acc_train = evaluate(train_loader_eval)
    acc_val = evaluate(val_loader_eval)

    # update lr
    scheduler.step()

    print(f'epoch {i_epoch + 1}: acc_train {acc_train:.3f}, acc_val {acc_val:.3f}')
    print(f' kl: {np.mean(kls)}, \nlog_likelihood: {np.mean(log_likelihoods)} '
          f'\nloss: {np.mean(losses)}')
