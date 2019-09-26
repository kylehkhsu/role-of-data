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
           dir='/scratch/hdd001/home/kylehsu/output/pacbayes_opt/data_dependent_prior_sgd/debug',
           tags=['debug'])

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/h/kylehsu/datasets')
parser.add_argument('--hidden_layer_sizes', type=list, nargs='+', default=[600])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--alpha', type=float, default=0.5)
parser.add_argument('--prior_and_posterior_mean_learning_rate', type=float, default=1e-2)
parser.add_argument('--posterior_variance_learning_rate', type=float, default=0.01)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--prior_var', type=float, default=0.003)
parser.add_argument('--min_prob', type=float, default=1e-4)
parser.add_argument('--delta', type=float, default=0.05)
parser.add_argument('--seed', type=int, default=43)
parser.add_argument('--posterior_mean_training_epochs', type=int, default=256)
parser.add_argument('--posterior_variance_training_epochs', type=int, default=256)
parser.add_argument('--prior_mean_training_epochs', type=int, default=256)
parser.add_argument('--prior_mean_patience', type=int, default=10)

args = parser.parse_args()
config = wandb.config
config.update(args)

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
prior_mean_train_set_size = int(config.alpha * train_set_size)
posterior_variance_train_set_size = train_set_size - prior_mean_train_set_size

prior_mean_train_set, posterior_variance_train_set = data.random_split(
    train_set, [prior_mean_train_set_size, posterior_variance_train_set_size]
)

posterior_mean_train_loader = data.DataLoader(
    dataset=train_set,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)
posterior_mean_train_loader_eval = data.DataLoader(
    dataset=train_set,
    batch_size=len(train_set),
    num_workers=2,
)
if config.alpha != 0:
    prior_mean_train_loader = data.DataLoader(
        dataset=prior_mean_train_set,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )
posterior_variance_train_loader = data.DataLoader(
    dataset=posterior_variance_train_set,
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

mlp_posterior_mean = MLP(
    n_input=784,
    n_output=10,
    hidden_layer_sizes=config.hidden_layer_sizes
)
classifier_posterior_mean = Classifier(
    net=mlp_posterior_mean
)
classifier_posterior_mean = classifier_posterior_mean.to(device)

posterior_mean_optimizer = torch.optim.SGD(
    classifier_posterior_mean.parameters(),
    lr=config.prior_and_posterior_mean_learning_rate,
    momentum=config.momentum
)
classifier_posterior_mean_init = deepcopy(classifier_posterior_mean).to(device)


def evaluate_classifier(classifier, loader):
    classifier.eval()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            probs = classifier(x)
            correct, total = classifier.evaluate(probs, y)
            error = 1 - correct / total
    return error


# train for one epoch on S_alpha
if config.alpha != 0:
    surrogate_bounds = []
    corrects, totals = 0.0, 0.0
    classifier_posterior_mean.train()
    for x, y in tqdm(prior_mean_train_loader, total=prior_mean_train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)
        probs = classifier_posterior_mean(x)
        loss = classifier_posterior_mean.loss(probs, y)
        posterior_mean_optimizer.zero_grad()
        loss.backward()
        posterior_mean_optimizer.step()

        with torch.no_grad():
            correct, total = classifier_posterior_mean.evaluate(probs, y)
        surrogate_bounds.append(loss.item())
        totals += total.item()
        corrects += correct.item()
    error_train = 1 - corrects / totals
    error_test = evaluate_classifier(classifier_posterior_mean, test_loader)
    log = {
        'loss_posterior_mean_train': np.mean(surrogate_bounds),
        'error_posterior_mean_train': error_train,
        'error_posterior_mean_test': error_test.item()
    }
    pp.pprint(log)

# coupling
classifier_prior_mean = deepcopy(classifier_posterior_mean).to(device)


def l2_between_mlps(mlp1, mlp2):
    mlp1_vector = torch.cat([p.view(-1) for p in mlp1.parameters() if p.requires_grad])
    mlp2_vector = torch.cat([p.view(-1) for p in mlp2.parameters() if p.requires_grad])
    return (mlp1_vector - mlp2_vector).norm()


# for the posterior mean, finish off the epoch of S with S \ S_alpha
surrogate_bounds = []
corrects, totals = 0.0, 0.0
classifier_posterior_mean.train()
for x, y in tqdm(posterior_variance_train_loader, total=posterior_variance_train_set_size // config.batch_size):
    x, y = x.to(device), y.to(device)
    probs = classifier_posterior_mean(x)
    loss = classifier_posterior_mean.loss(probs, y)
    posterior_mean_optimizer.zero_grad()
    loss.backward()
    posterior_mean_optimizer.step()

    with torch.no_grad():
        correct, total = classifier_posterior_mean.evaluate(probs, y)
    surrogate_bounds.append(loss.item())
    totals += total.item()
    corrects += correct.item()
error_train = 1 - corrects / totals
error_test = evaluate_classifier(classifier_posterior_mean, test_loader)
log = {
    'loss_posterior_mean_train': np.mean(surrogate_bounds),
    'error_posterior_mean_train': error_train,
    'error_posterior_mean_test': error_test.item()
}
pp.pprint(log)

# finish optimizing posterior mean on S
for i_epoch in tqdm(range(config.posterior_mean_training_epochs)):
    classifier_posterior_mean.train()
    surrogate_bounds = []
    corrects, totals = 0.0, 0.0
    for x, y in tqdm(posterior_mean_train_loader, total=train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)
        probs = classifier_posterior_mean(x)
        loss = classifier_posterior_mean.loss(probs, y)
        posterior_mean_optimizer.zero_grad()
        loss.backward()
        posterior_mean_optimizer.step()

        with torch.no_grad():
            correct, total = classifier_posterior_mean.evaluate(probs, y)
        surrogate_bounds.append(loss.item())
        totals += total.item()
        corrects += correct.item()

    error_train_running = 1 - corrects / totals
    error_test = evaluate_classifier(classifier_posterior_mean, test_loader)
    error_train_end = evaluate_classifier(classifier_posterior_mean, posterior_mean_train_loader_eval)

    log = {
        'epoch_posterior_mean': i_epoch,
        'loss_posterior_mean_train': np.mean(surrogate_bounds),
        'error_posterior_mean_train': error_train_running,
        'error_posterior_mean_test': error_test.item(),
        'error_posterior_mean_train_end': error_train_end.item()
    }
    pp.pprint(log)

    if error_train_end.item() <= 0.01:
        print('exited posterior mean training due to epoch train accuracy exceeding 0.99')
        break

print(f'l2 distance between trained posterior mean and initial posterior mean: {l2_between_mlps(classifier_posterior_mean.net, classifier_posterior_mean_init.net)}')
print(f'l2 distance between trained posterior mean and initial prior mean: {l2_between_mlps(classifier_posterior_mean.net,classifier_prior_mean.net)}')
print(f'l2 distance between initial posterior mean and initial prior mean: {l2_between_mlps(classifier_posterior_mean_init.net, classifier_prior_mean.net)}')

prior_mean_optimizer = torch.optim.SGD(
    classifier_prior_mean.parameters(),
    lr=config.prior_and_posterior_mean_learning_rate,
    momentum=config.momentum
)
l2_closest = float('inf')
classifier_prior_mean_closest = deepcopy(classifier_prior_mean)
i_epoch_closest = 0
if config.alpha != 0:
    for i_epoch in tqdm(range(config.prior_mean_training_epochs)):
        classifier_prior_mean.train()
        surrogate_bounds = []
        corrects, totals = 0.0, 0.0
        for x, y in tqdm(prior_mean_train_loader, total=prior_mean_train_set_size // config.batch_size):
            x, y = x.to(device), y.to(device)
            probs = classifier_prior_mean(x)
            loss = classifier_prior_mean.loss(probs, y)
            prior_mean_optimizer.zero_grad()
            loss.backward()
            prior_mean_optimizer.step()

            with torch.no_grad():
                correct, total = classifier_prior_mean.evaluate(probs, y)
            surrogate_bounds.append(loss.item())
            totals += total.item()
            corrects += correct.item()
        error_train = 1 - corrects / totals
        error_test = evaluate_classifier(classifier_prior_mean, test_loader)
        with torch.no_grad():
            l2 = l2_between_mlps(classifier_prior_mean.net, classifier_posterior_mean.net).item()
        if l2 < l2_closest:
            classifier_prior_mean_closest = deepcopy(classifier_prior_mean)
            l2_closest = l2
            i_epoch_closest = i_epoch

        log = {
            'epoch_prior_mean': i_epoch,
            'loss_prior_mean_train': np.mean(surrogate_bounds),
            'error_prior_mean_train': error_train,
            'error_prior_mean_test': error_test.item(),
            'l2_prior_to_posterior': l2,
            'l2_closest': l2_closest
        }
        pp.pprint(log)
        if i_epoch - i_epoch_closest > config.prior_mean_patience:
            break

bayesian_classifier = make_bayesian_classifier_from_mlps(
    mlp_posterior_mean_init=classifier_posterior_mean.net,
    mlp_prior_mean=classifier_prior_mean_closest.net,
    prior_stddev=math.sqrt(config.prior_var),
    optimize_prior_mean=False,
    optimize_prior_rho=False,
    optimize_posterior_mean=False,
    optimize_posterior_rho=True,
    probability_threshold=config.min_prob,
    normalize_surrogate_by_log_classes=True
)

bayesian_classifier = bayesian_classifier.to(device)
posterior_variance_optimizer = torch.optim.SGD(
    bayesian_classifier.parameters(),
    lr=config.posterior_variance_learning_rate,
    momentum=config.momentum
)


def evaluate_bayesian_classifier(bayesian_classifier):
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


for i_epoch in tqdm(range(config.posterior_variance_training_epochs)):
    bayesian_classifier.train()
    kls, surrogates, surrogate_bounds, error_bounds = [], [], [], []
    corrects, totals = 0.0, 0.0
    for x, y in tqdm(posterior_variance_train_loader, total=posterior_variance_train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)
        x = x.view([x.shape[0], -1])

        kl, surrogate, correct = bayesian_classifier.forward_train(x, y)
        surrogate_bound = bayesian_classifier.inverted_kl_bound(surrogate, kl, posterior_variance_train_set_size,
                                                                config.delta)
        posterior_variance_optimizer.zero_grad()
        surrogate_bound.backward()
        posterior_variance_optimizer.step()

        total = y.shape[0]
        error = 1 - correct / total
        with torch.no_grad():
            error_bound = bayesian_classifier.inverted_kl_bound(error, kl, posterior_variance_train_set_size,
                                                                config.delta)

        totals += total
        corrects += correct.item()
        kls.append(kl.item())
        surrogates.append(surrogate.item())
        surrogate_bounds.append(surrogate_bound.item())
        error_bounds.append(error_bound.item())

    error_test = evaluate_bayesian_classifier(bayesian_classifier)

    log = {
        'error_bound': np.mean(error_bounds),
        'surrogate_bound': np.mean(surrogate_bounds),
        'error_train': 1 - corrects / totals,
        'kl_normalized': np.mean(kls) / posterior_variance_train_set_size,
        'surrogate_risk': np.mean(surrogates),
        'error_test': error_test
    }
    wandb.log(log)
    pp.pprint(log)

torch.save(bayesian_classifier.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
