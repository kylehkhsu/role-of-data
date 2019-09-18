import torch
import torchvision
import ipdb
import numpy as np
import torch.utils.data as data
import math

import wandb
import os
from src.model.pacbayes_by_backprop import make_bnn_mlp

from tqdm import tqdm

wandb.init(project="pacbayes_opt", dir='/h/kylehsu/output/pacbayes_opt/mnist/quad_bound/exp001')

dataset_path = '/h/kylehsu/datasets'

torch.manual_seed(42)
np.random.seed(43)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

config_defaults = dict(
    batch_size=256,
    n_samples=1,
    n_epochs=256,
    learning_rate=0.01,
    momentum=0.99,
    prior_var=0.1,
    hidden_layer_sizes=[600, 600],
    min_prob=1e-4,
    delta=0.05,
)
config = wandb.config
config.update({k: v for k, v in config_defaults.items() if k not in dict(config.user_items())})

train_and_val_set = torchvision.datasets.MNIST(
    root=dataset_path,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_set_size = 50000

train_set, val_set = data.random_split(
    train_and_val_set, [train_set_size, len(train_and_val_set) - train_set_size])
train_loader = data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True, drop_last=True)
train_loader_eval = data.DataLoader(train_set, batch_size=len(train_set))
val_loader_eval = data.DataLoader(val_set, batch_size=len(val_set))

bnn = make_bnn_mlp(
    n_input=784,
    n_output=10,
    hidden_layer_sizes=config.hidden_layer_sizes,
    prior_std=math.sqrt(config.prior_var),
    min_prob=config.min_prob
)
wandb.watch(bnn)
bnn = bnn.to(device)

optim = torch.optim.SGD(
    bnn.parameters(),
    lr=config.learning_rate,
    momentum=config.momentum
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


# def lambda_bound(risk, kl, dataset_size, delta, lam):
#     raise NotImplementedError
#     log_2_sqrt_n_over_delta = math.log(2 * math.sqrt(dataset_size) / delta)
#     term1 = risk.div(1 - lam / 2)
#     term2 = (kl + log_2_sqrt_n_over_delta).div(dataset_size * lam * (1 - lam / 2))
#     return term1 + term2

bound = quad_bound

for i_epoch in tqdm(range(config.n_epochs)):
    bnn.train()
    kls = []
    log_likelihoods = []
    losses = []
    corrects = 0
    totals = 0
    for x, y in tqdm(train_loader, total=len(train_set) // config.batch_size):
        x, y = x.to(device), y.to(device)

        kl, log_likelihood, correct = bnn.forward_train(x, y, config.n_samples)
        loss = bound(-log_likelihood, kl, train_set_size, config.delta)
        optim.zero_grad()
        loss.backward()
        optim.step()

        totals += y.shape[0]
        corrects += correct.item()
        kls.append(kl.item())
        log_likelihoods.append(log_likelihood.item())
        losses.append(loss.item())

    # eval
    acc_val = evaluate(val_loader_eval)

    log = {
        'loss_train': np.mean(losses),
        'err_train': 1 - corrects / totals,
        'kl_normalized_train': np.mean(kls) / train_set_size,
        'risk_train': -np.mean(log_likelihoods),
        'learning_rate': scheduler.get_lr()[0],
        'err_val': 1 - acc_val
    }
    wandb.log(log)

    # update lr
    scheduler.step()

torch.save(bnn.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
