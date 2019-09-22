import torch
import torchvision
import torch.utils.data as data
import argparse
import numpy as np
import ipdb
from src.model.mlp import MLP
from src.model.pacbayes_by_backprop import make_bmlp_from_mlps, BMLP
from tqdm import tqdm
import pprint
from copy import deepcopy
import wandb
import math
import os

pp = pprint.PrettyPrinter()
wandb.init(project="pacbayes_opt",
           dir='/scratch/hdd001/home/kylehsu/output/pacbayes_opt/data_dependent_prior/')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/h/kylehsu/datasets')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--prior_learning_rate', type=float, default=1e-2)
parser.add_argument('--posterior_learning_rate', type=float, default=1e-2)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_layer_sizes', type=list, nargs='+', default=[600, 600])
parser.add_argument('--prior_training_epochs', type=int, default=10)
parser.add_argument('--posterior_training_epochs', type=int, default=256)
parser.add_argument('--prior_var', type=float, default=0.001)
parser.add_argument('--min_prob', type=float, default=1e-4)
parser.add_argument('--delta', type=float, default=0.05)
parser.add_argument('--reparam_trick', type=str, default='global')
parser.add_argument('--debug', action='store_true')
parser.add_argument('--prior_seed', type=int, default=42)
parser.add_argument('--posterior_seed', type=int, default=43)

args = parser.parse_args()
config = wandb.config
config.update(args)

if config.debug:
    os.environ['WANDB_MODE'] = 'dryrun'

torch.manual_seed(config.prior_seed)
np.random.seed(config.prior_seed)

train_set = torchvision.datasets.MNIST(
    root=config.dataset_path,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_set = torchvision.datasets.MNIST(
    root=config.dataset_path,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_set_size = len(train_set)
prior_train_set_size = int(config.alpha * train_set_size)
posterior_train_set_size = train_set_size - prior_train_set_size

prior_train_set, posterior_train_set = data.random_split(
    train_set, [prior_train_set_size, posterior_train_set_size]
)
print(
    f'alpha: {config.alpha}',
    f'prior_train_set_size: {prior_train_set_size}',
    f'posterior_train_set_size: {posterior_train_set_size}'
)

if config.alpha != 0:
    prior_train_loader = data.DataLoader(
        dataset=prior_train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True
    )
posterior_train_loader = data.DataLoader(
    dataset=posterior_train_set,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True
)
test_loader = data.DataLoader(
    dataset=test_set,
    batch_size=len(test_set)
)

mlp = MLP(
    n_input=784,
    n_output=10,
    hidden_layer_sizes=config.hidden_layer_sizes
)
mlp = mlp.to(device)

prior_optimizer = torch.optim.SGD(
    mlp.parameters(),
    lr=config.prior_learning_rate,
    momentum=config.momentum
)


def evaluate_mlp():
    mlp.eval()
    with torch.no_grad():
        for x, y in test_loader:
            assert y.shape[0] == len(test_set)
            x, y = x.to(device), y.to(device)
            probs = mlp(x)
            correct, total = mlp.evaluate(probs, y)
            error = 1 - correct / total
    return error


mlp_posterior_mean_init = None

if config.alpha != 0:
    for i_epoch in tqdm(range(config.prior_training_epochs)):
        mlp.train()
        losses = []
        corrects, totals = 0.0, 0.0
        for x, y in tqdm(prior_train_loader, total=prior_train_set_size // config.batch_size):
            x, y = x.to(device), y.to(device)

            probs = mlp.forward(x)
            loss = mlp.loss(probs, y)
            prior_optimizer.zero_grad()
            loss.backward()
            prior_optimizer.step()

            with torch.no_grad():
                correct, total = mlp.evaluate(probs, y)

            losses.append(loss.item())
            totals += total.item()
            corrects += correct.item()

        if i_epoch == 0:
            mlp_posterior_mean_init = deepcopy(mlp).to('cpu')
            print('saved copy of mlp as mlp_posterior_mean_init')

        error_train = 1 - corrects / totals
        error_test = evaluate_mlp()

        log = {
            'prior_training_epoch': i_epoch,
            'loss_prior_train': np.mean(losses),
            'error_train': error_train,
            'error_test': error_test.item()
        }
        pp.pprint(log)

        if error_train < 1e-5:
            break
else:
    log = {
        'error_test': evaluate_mlp().item()
    }
    pp.pprint(log)
mlp_prior = mlp.to('cpu')
if mlp_posterior_mean_init is None:
    assert config.alpha == 0 or config.prior_training_epochs == 0
    print('no prior was trained')
    mlp_posterior_mean_init = deepcopy(mlp).to('cpu')

torch.manual_seed(config.posterior_seed)
np.random.seed(config.posterior_seed)

bmlp = make_bmlp_from_mlps(
    mlp_posterior_mean_init=mlp_posterior_mean_init,
    mlp_prior=mlp_prior,
    prior_std=math.sqrt(config.prior_var),
    min_prob=config.min_prob,
    reparam_trick=config.reparam_trick
)
bmlp = bmlp.to(device)
posterior_optimizer = torch.optim.SGD(
    bmlp.parameters(),
    lr=config.posterior_learning_rate,
    momentum=config.momentum
)


def evaluate_bmlp():
    bmlp.eval()
    with torch.no_grad():
        for x, y in test_loader:
            assert y.shape[0] == len(test_set)
            x, y = x.to(device), y.to(device)
            probs = bmlp(x, 'MC')
            preds = probs.argmax(dim=-1)
            error = 1 - (y == preds).sum().float().div(y.shape[0]).item()
    return error


for i_epoch in tqdm(range(config.posterior_training_epochs)):
    bmlp.train()
    kls, log_likelihoods, losses, bounds = [], [], [], []
    corrects, totals = 0.0, 0.0
    for x, y in tqdm(posterior_train_loader, total=posterior_train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)

        kl, log_likelihood, correct = bmlp.forward_train(x, y)
        loss = bmlp.quad_bound(-log_likelihood, kl, posterior_train_set_size, config.delta)
        posterior_optimizer.zero_grad()
        loss.backward()
        posterior_optimizer.step()

        total = y.shape[0]
        error = 1 - correct / total
        with torch.no_grad():
            bound = bmlp.quad_bound(error, kl, posterior_train_set_size, config.delta)

        totals += total
        corrects += correct.item()
        kls.append(kl.item())
        log_likelihoods.append(log_likelihood.item())
        losses.append(loss.item())
        bounds.append(bound.item())

    error_test = evaluate_bmlp()

    log = {
        'error_bound': np.mean(bounds),
        'loss_train': np.mean(losses),
        'error_train': 1 - corrects / totals,
        'kl_normalized_train': np.mean(kls) / posterior_train_set_size,
        'risk_surrogate_train': -np.mean(log_likelihoods),
        'error_test': error_test
    }
    wandb.log(log)
    pp.pprint(log)

torch.save(bmlp.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
