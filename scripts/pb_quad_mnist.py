import torch
import torchvision
import ipdb
import numpy as np
import torch.utils.data as data
import math
from src.model.makers import make_bayesian_mlp_classifier
import wandb
import os
import argparse
import pprint
from tqdm import tqdm

pp = pprint.PrettyPrinter()
wandb.init(
    project="pacbayes_opt",
    dir='/scratch/hdd001/home/kylehsu/output/pacbayes_opt/quad_bound/mnist/debug',
    tags=['debug']
)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/h/kylehsu/datasets')
parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_layer_sizes', type=list, nargs='+', default=[600]*2)
parser.add_argument('--n_epochs', type=int, default=256)
parser.add_argument('--prior_var', type=float, default=0.01)
parser.add_argument('--min_prob', type=float, default=1e-4)
parser.add_argument('--delta', type=float, default=0.05)
parser.add_argument('--debug', action='store_true')
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()
config = wandb.config
config.update(args)

if config.debug:
    os.environ['WANDB_MODE'] = 'dryrun'

torch.manual_seed(config.seed)
np.random.seed(config.seed)

train_and_val_set = torchvision.datasets.MNIST(
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

train_set_size = 50000

train_set, val_set = data.random_split(
    train_and_val_set, [train_set_size, len(train_and_val_set) - train_set_size])
train_loader = data.DataLoader(
    dataset=train_set,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2
)
test_loader = data.DataLoader(
    dataset=test_set,
    batch_size=len(test_set),
    num_workers=2
)

bayesian_classifier = make_bayesian_mlp_classifier(
    n_input=784,
    n_output=10,
    hidden_layer_sizes=config.hidden_layer_sizes,
    prior_stddev=math.sqrt(config.prior_var),
    optimize_prior_mean=False,
    optimize_prior_rho=False,
    optimize_posterior_mean=True,
    optimize_posterior_rho=True,
    probability_threshold=config.min_prob,
    normalize_surrogate_by_log_classes=True
)
bayesian_classifier = bayesian_classifier.to(device)

optim = torch.optim.SGD(
    bayesian_classifier.parameters(),
    lr=config.learning_rate,
    momentum=config.momentum
)


def evaluate_bayesian_classifier(loader):
    bayesian_classifier.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            x = x.view([x.shape[0], -1])
            out = bayesian_classifier(x, 'MC')
            y_pred = out.argmax(dim=1)
            accuracy = (y == y_pred).sum().float().div(y.shape[0]).item()
        return accuracy


for i_epoch in tqdm(range(config.n_epochs)):
    bayesian_classifier.train()
    kls = []
    surrogates = []
    losses = []
    bounds = []
    corrects = 0
    totals = 0
    for x, y in tqdm(train_loader, total=len(train_set) // config.batch_size):
        x, y = x.to(device), y.to(device)
        x = x.view([x.shape[0], -1])

        kl, surrogate, correct = bayesian_classifier.forward_train(x, y)
        loss = bayesian_classifier.quad_bound(surrogate, kl, train_set_size, config.delta)
        optim.zero_grad()
        loss.backward()
        optim.step()

        total = y.shape[0]
        error = 1 - correct / total
        with torch.no_grad():
            bound = bayesian_classifier.quad_bound(error, kl, train_set_size, config.delta)

        totals += total
        corrects += correct.item()
        kls.append(kl.item())
        surrogates.append(surrogate.item())
        losses.append(loss.item())
        bounds.append(bound.item())

    # eval
    acc_test = evaluate_bayesian_classifier(test_loader)

    log = {
        'error_bound': np.mean(bounds),
        'loss_train': np.mean(losses),
        'error_train': 1 - corrects / totals,
        'kl_normalized_train': np.mean(kls) / train_set_size,
        'risk_surrogate_train': np.mean(surrogates),
        'error_test': 1 - acc_test
    }
    pp.pprint(log)
    wandb.log(log)

# torch.save(bayesian_classifier.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
