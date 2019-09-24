import torch
import torchvision
import torch.utils.data as data
import argparse
import numpy as np
import ipdb
from src.model.mlp import MLP
from src.model.classifier import Classifier
from src.model.makers import make_bayesian_classifier_from_mlps
from tqdm import tqdm
import pprint
from copy import deepcopy
import wandb
import math
import os

pp = pprint.PrettyPrinter()
wandb.init(project="pacbayes_opt",
           dir='/scratch/hdd001/home/kylehsu/output/pacbayes_opt/data_dependent_prior/',
           tags=['debug'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/h/kylehsu/datasets')
parser.add_argument('--alpha', type=float, default=0.1)
parser.add_argument('--prior_learning_rate', type=float, default=1e-2)
parser.add_argument('--posterior_learning_rate', type=float, default=0.00001)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--hidden_layer_sizes', type=list, nargs='+', default=[600] * 3)
parser.add_argument('--prior_training_epochs', type=int, default=100)
parser.add_argument('--posterior_training_epochs', type=int, default=256)
parser.add_argument('--prior_var', type=float, default=1e-7)
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
        drop_last=True,
        num_workers=2
    )
posterior_train_loader = data.DataLoader(
    dataset=posterior_train_set,
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

mlp = MLP(
    n_input=784,
    n_output=10,
    hidden_layer_sizes=config.hidden_layer_sizes
)
classifier = Classifier(
    net=mlp
)
classifier = classifier.to(device)

prior_optimizer = torch.optim.SGD(
    classifier.parameters(),
    lr=config.prior_learning_rate,
    momentum=config.momentum
)


def evaluate_classifier():
    classifier.eval()
    with torch.no_grad():
        for x, y in test_loader:
            assert y.shape[0] == len(test_set)
            x, y = x.to(device), y.to(device)
            probs = classifier(x)
            correct, total = classifier.evaluate(probs, y)
            error = 1 - correct / total
    return error


classifier_posterior_mean_init = None

if config.alpha != 0:
    for i_epoch in tqdm(range(config.prior_training_epochs)):
        classifier.train()
        losses = []
        corrects, totals = 0.0, 0.0
        for x, y in tqdm(prior_train_loader, total=prior_train_set_size // config.batch_size):
            x, y = x.to(device), y.to(device)

            probs = classifier(x)
            loss = classifier.loss(probs, y)
            prior_optimizer.zero_grad()
            loss.backward()
            prior_optimizer.step()

            with torch.no_grad():
                correct, total = classifier.evaluate(probs, y)

            losses.append(loss.item())
            totals += total.item()
            corrects += correct.item()

        if i_epoch == 0:
            classifier_posterior_mean_init = deepcopy(classifier).to('cpu')
            print('saved copy of classifier as classifier_posterior_mean_init')

        error_train = 1 - corrects / totals
        error_test = evaluate_classifier()

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
        'error_test': evaluate_classifier().item()
    }
    pp.pprint(log)
classifier_prior_mean = classifier.to('cpu')
if classifier_posterior_mean_init is None:
    assert config.alpha == 0 or config.prior_training_epochs == 0
    print('no prior was trained')
    classifier_posterior_mean_init = deepcopy(classifier).to('cpu')

bayesian_classifier = make_bayesian_classifier_from_mlps(
    mlp_posterior_mean_init=classifier_posterior_mean_init.net,
    mlp_prior_mean=classifier_prior_mean.net,
    prior_stddev=math.sqrt(config.prior_var),
    optimize_prior_mean=False,
    optimize_prior_rho=False,
    optimize_posterior_mean=True,
    optimize_posterior_rho=True,
    probability_threshold=config.min_prob,
    normalize_surrogate_by_log_classes=True
)
bayesian_classifier = bayesian_classifier.to(device)
posterior_optimizer = torch.optim.SGD(
    bayesian_classifier.parameters(),
    lr=config.posterior_learning_rate,
    momentum=config.momentum
)


def evaluate_bayesian_classifier():
    bayesian_classifier.eval()
    with torch.no_grad():
        for x, y in test_loader:
            assert y.shape[0] == len(test_set)
            x, y = x.to(device), y.to(device)
            x = x.view([x.shape[0], -1])
            probs = bayesian_classifier(x, 'MC')
            preds = probs.argmax(dim=-1)
            error = 1 - (y == preds).sum().float().div(y.shape[0]).item()
    return error


for i_epoch in tqdm(range(config.posterior_training_epochs)):
    bayesian_classifier.train()
    kls, surrogates, surrogate_bounds, error_bounds = [], [], [], []
    corrects, totals = 0.0, 0.0
    for x, y in tqdm(posterior_train_loader, total=posterior_train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)
        x = x.view([x.shape[0], -1])

        kl, surrogate, correct = bayesian_classifier.forward_train(x, y)
        surrogate_bound = bayesian_classifier.quad_bound(surrogate, kl, posterior_train_set_size, config.delta)
        posterior_optimizer.zero_grad()
        surrogate_bound.backward()
        posterior_optimizer.step()

        total = y.shape[0]
        error = 1 - correct / total
        with torch.no_grad():
            error_bound = bayesian_classifier.quad_bound(error, kl, posterior_train_set_size, config.delta)

        totals += total
        corrects += correct.item()
        kls.append(kl.item())
        surrogates.append(surrogate.item())
        surrogate_bounds.append(surrogate_bound.item())
        error_bounds.append(error_bound.item())

    error_test = evaluate_bayesian_classifier()

    log = {
        'error_bound': np.mean(error_bounds),
        'surrogate_bound': np.mean(surrogate_bounds),
        'error_train': 1 - corrects / totals,
        'kl_normalized': np.mean(kls) / posterior_train_set_size,
        'surrogate_risk': np.mean(surrogates),
        'error_test': error_test
    }
    wandb.log(log)
    pp.pprint(log)

torch.save(bayesian_classifier.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
