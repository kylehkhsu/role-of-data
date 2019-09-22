import torch
import torch.nn as nn
import torchvision
import ipdb
import numpy as np
import torch.utils.data as data
import math

import wandb
import os
from src.model.pacbayes_by_backprop import make_bnn_mlp, BMLP

from tqdm import tqdm

wandb.init(project="pacbayes_opt", dir='/scratch/hdd001/home/kylehsu/output/pacbayes_opt/mnist/lambda_bound/debug')

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
    reparam_trick='global',
    covariance_init_strategy='isotropic',
    mean_init_strategy='use_prior_std',
    lambda_init=1.0,
    lambda_learning_rate=1e-4
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
    min_prob=config.min_prob,
    reparam_trick=config.reparam_trick,
    config=config
)

wandb.watch(bnn)
bnn = bnn.to(device)
optim = torch.optim.SGD(
    bnn.parameters(),
    lr=config.learning_rate,
    momentum=config.momentum
)
scheduler = torch.optim.lr_scheduler.StepLR(optim, step_size=100, gamma=1)

# the λ value start at 1 and decreases to 0.5 and starts increasing again after around 150000 iterations to
# finally reach a value of 0.75.
# The lambda value in flamb was optimized using
# alternate minimization using SGD with fixed learning rate of 1e − 4
lam = nn.Module()
lam.register_parameter('lam', nn.Parameter(torch.ones([]) * config.lambda_init))
lam.to(device)
lam_optim = torch.optim.SGD(
    lam.parameters(),
    lr=config.lambda_learning_rate
)


def evaluate(loader):
    bnn.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = bnn(x, 'MAP')
            y_pred = out.argmax(dim=1)
            accuracy = (y == y_pred).sum().float().div(y.shape[0]).item()
        return accuracy


for i_epoch in tqdm(range(config.n_epochs)):
    bnn.train()
    kls = []
    log_likelihoods = []
    losses = []
    bounds = []
    lams = []
    corrects = 0
    totals = 0
    for x, y in tqdm(train_loader, total=len(train_set) // config.batch_size):
        x, y = x.to(device), y.to(device)

        kl, log_likelihood, _ = bnn.forward_train(x, y, config.n_samples)
        loss = BMLP.lambda_bound(-log_likelihood, kl, train_set_size, config.delta, lam.lam)
        optim.zero_grad()
        loss.backward()
        optim.step()

        kl, log_likelihood, correct = bnn.forward_train(x, y, config.n_samples)
        loss = BMLP.lambda_bound(-log_likelihood, kl, train_set_size, config.delta, lam.lam)
        lam_optim.zero_grad()
        loss.backward()
        lam_optim.step()

        total = y.shape[0]
        error = 1 - correct / total
        with torch.no_grad():
            bound = BMLP.lambda_bound(error, kl, train_set_size, config.delta, lam.lam)

        totals += total
        corrects += correct.item()
        kls.append(kl.item())
        log_likelihoods.append(log_likelihood.item())
        losses.append(loss.item())
        bounds.append(bound.item())
        lams.append(lam.lam.item())

    # eval
    acc_val = evaluate(val_loader_eval)

    log = {
        'error_bound': np.mean(bounds),
        'loss_train': np.mean(losses),
        'error_train': 1 - corrects / totals,
        'kl_normalized_train': np.mean(kls) / train_set_size,
        'risk_surrogate_train': -np.mean(log_likelihoods),
        'learning_rate': scheduler.get_lr()[0],
        'error_val': 1 - acc_val,
        'lambda': np.mean(lams)
    }
    wandb.log(log)

    # update lr
    scheduler.step()

torch.save(bnn.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
