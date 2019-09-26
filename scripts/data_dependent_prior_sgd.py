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
parser.add_argument('--posterior_mean_stopping_error_train', type=float, default=0.01)

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
train_set_size_prior_mean = int(config.alpha * train_set_size)
train_set_size_posterior_variance = train_set_size - train_set_size_prior_mean

train_set_prior_mean, train_set_posterior_variance = data.random_split(
    train_set, [train_set_size_prior_mean, train_set_size_posterior_variance]
)

# S, the entire training set
train_loader_posterior_mean = data.DataLoader(
    dataset=train_set,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2,
)

# S but with large batch size; for use in eval (no gradients)
train_loader_eval = data.DataLoader(
    dataset=train_set,
    batch_size=len(train_set)//10,
    num_workers=2,
)

if config.alpha != 0:
    # S_alpha
    train_loader_prior_mean = data.DataLoader(
        dataset=train_set_prior_mean,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=2
    )

# S \ S_alpha
train_loader_posterior_variance = data.DataLoader(
    dataset=train_set_posterior_variance,
    batch_size=config.batch_size,
    shuffle=True,
    drop_last=True,
    num_workers=2
)
train_loader_eval_posterior_variance = data.DataLoader(
    dataset=train_set_posterior_variance,
    batch_size=train_set_size_posterior_variance // 10,
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

optimizer_posterior_mean = torch.optim.SGD(
    classifier_posterior_mean.parameters(),
    lr=config.prior_and_posterior_mean_learning_rate,
    momentum=config.momentum
)
classifier_posterior_mean_init = deepcopy(classifier_posterior_mean).to(device)


def train_classifier_epoch(
        classifier,
        optimizer,
        train_loader,
        train_set_size,
        train_loader_eval,
        test_loader
):
    training = classifier.training

    classifier.train()
    for x, y in tqdm(train_loader, total=train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)
        probs = classifier(x)
        cross_entropy = classifier.cross_entropy(probs, y)
        optimizer.zero_grad()
        cross_entropy.backward()
        optimizer.step()

    error_train, cross_entropy_train = classifier.evaluate_on_loader(train_loader_eval)
    error_test, cross_entropy_test = classifier.evaluate_on_loader(test_loader)

    log = {
        'cross_entropy_train': cross_entropy_train,
        'cross_entropy_test': cross_entropy_test,
        'error_train': error_train,
        'error_test': error_test
    }

    classifier.train(mode=training)

    return log

# train for one epoch on S_alpha; coupling
if config.alpha != 0:
    print('optimizing posterior and prior means for one epoch of S_alpha')
    log = train_classifier_epoch(
        classifier=classifier_posterior_mean,
        optimizer=optimizer_posterior_mean,
        train_loader=train_loader_prior_mean,
        train_set_size=train_set_size_prior_mean,
        train_loader_eval=train_loader_eval,
        test_loader=test_loader
    )
    pp.pprint(log)

# coupling
classifier_prior_mean = deepcopy(classifier_posterior_mean).to(device)


def l2_between_mlps(mlp1, mlp2):
    mlp1_vector = torch.cat([p.view(-1) for p in mlp1.parameters() if p.requires_grad])
    mlp2_vector = torch.cat([p.view(-1) for p in mlp2.parameters() if p.requires_grad])
    return (mlp1_vector - mlp2_vector).norm()


# for the posterior mean, finish off the epoch of S with S \ S_alpha
print('optimizing posterior mean for one epoch of S \ S_alpha')
log = train_classifier_epoch(
    classifier=classifier_posterior_mean,
    optimizer=optimizer_posterior_mean,
    train_loader=train_loader_posterior_variance,
    train_set_size=train_set_size_posterior_variance,
    train_loader_eval=train_loader_eval,
    test_loader=test_loader
)
pp.pprint(log)

# finish optimizing posterior mean on S
print(f'optimizing posterior mean to error_train {config.posterior_mean_stopping_error_train}')
for i_epoch in tqdm(range(1, config.posterior_mean_training_epochs)):
    log = train_classifier_epoch(
        classifier=classifier_posterior_mean,
        optimizer=optimizer_posterior_mean,
        train_loader=train_loader_posterior_mean,
        train_set_size=train_set_size,
        train_loader_eval=train_loader_eval,
        test_loader=test_loader
    )
    log.update({
        'epoch_posterior_mean': i_epoch
    })
    pp.pprint(log)

    if log['error_train'] <= config.posterior_mean_stopping_error_train:
        break

print(f'l2 distance between trained posterior mean and initial posterior mean: {l2_between_mlps(classifier_posterior_mean.net, classifier_posterior_mean_init.net)}')
print(f'l2 distance between trained posterior mean and initial prior mean: {l2_between_mlps(classifier_posterior_mean.net,classifier_prior_mean.net)}')
print(f'l2 distance between initial posterior mean and initial prior mean: {l2_between_mlps(classifier_posterior_mean_init.net, classifier_prior_mean.net)}')

optimizer_prior_mean = torch.optim.SGD(
    classifier_prior_mean.parameters(),
    lr=config.prior_and_posterior_mean_learning_rate,
    momentum=config.momentum
)
l2_closest = float('inf')
classifier_prior_mean_closest = deepcopy(classifier_prior_mean)
i_epoch_closest = 0
if config.alpha != 0:
    for i_epoch in tqdm(range(config.prior_mean_training_epochs)):
        log = train_classifier_epoch(
            classifier=classifier_prior_mean,
            optimizer=optimizer_prior_mean,
            train_loader=train_loader_prior_mean,
            train_set_size=train_set_size_prior_mean,
            train_loader_eval=train_loader_eval,
            test_loader=test_loader
        )

        with torch.no_grad():
            l2 = l2_between_mlps(classifier_prior_mean.net, classifier_posterior_mean.net).item()
        if l2 < l2_closest:
            classifier_prior_mean_closest = deepcopy(classifier_prior_mean)
            l2_closest = l2
            i_epoch_closest = i_epoch

        log.update({
            'epoch_prior_mean': i_epoch,
            'l2_prior_to_posterior': l2,
            'l2_prior_to_posterior_closest': l2_closest
        })
        pp.pprint(log)
        if i_epoch - i_epoch_closest > config.prior_mean_patience:
            break

bayesian_classifier = make_bayesian_classifier_from_mlps(
    mlp_posterior_mean_init=classifier_posterior_mean.net,
    mlp_prior_mean=classifier_prior_mean.net,
    prior_stddev=math.sqrt(config.prior_var),
    optimize_prior_mean=False,
    optimize_prior_rho=False,
    optimize_posterior_mean=False,
    optimize_posterior_rho=True,
    probability_threshold=config.min_prob,
    normalize_surrogate_by_log_classes=True
)

bayesian_classifier = bayesian_classifier.to(device)
optimizer_posterior_variance = torch.optim.SGD(
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


def train_bayesian_classifier_epoch(
        bayesian_classifier,
        optimizer,
        train_loader,
        train_set_size,
        train_loader_eval,
        test_loader
):
    training = bayesian_classifier.training
    bayesian_classifier.train()

    for x, y in tqdm(train_loader, total=train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)
        x = x.view([x.shape[0], -1])

        kl, surrogate = bayesian_classifier.forward_train(x, y)
        surrogate_bound = bayesian_classifier.inverted_kl_bound(
            risk=surrogate,
            kl=kl,
            dataset_size=train_set_size,
            delta=config.delta
        )
        optimizer.zero_grad()
        surrogate_bound.backward()
        optimizer.step()
        kl_last = kl.clone().detach()

    error_train, surrogate_train = bayesian_classifier.evaluate_on_loader(train_loader_eval)
    error_test, surrogate_test = bayesian_classifier.evaluate_on_loader(test_loader)
    with torch.no_grad():
        error_bound = bayesian_classifier.inverted_kl_bound(
            risk=error_train,
            kl=kl_last,
            dataset_size=train_set_size,
            delta=config.delta
        )
        surrogate_bound = bayesian_classifier.inverted_kl_bound(
            risk=surrogate_train,
            kl=kl_last,
            dataset_size=train_set_size,
            delta=config.delta
        )

    log = {
        'error_bound': error_bound.item(),
        'error_train': error_train,
        'error_test': error_test,
        'surrogate_bound': surrogate_bound.item(),
        'surrogate_train': surrogate_train,
        'surrogate_test': surrogate_test,
        'kl_normalized': kl_last.item() / train_set_size
    }

    bayesian_classifier.train(mode=training)

    return log
    

for i_epoch in tqdm(range(config.posterior_variance_training_epochs)):
    log = train_bayesian_classifier_epoch(
        bayesian_classifier=bayesian_classifier,
        optimizer=optimizer_posterior_variance,
        train_loader=train_loader_posterior_variance,
        train_set_size=train_set_size_posterior_variance,
        train_loader_eval=train_loader_eval_posterior_variance,
        test_loader=test_loader
    )

    wandb.log(log)
    pp.pprint(log)

torch.save(bayesian_classifier.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))
