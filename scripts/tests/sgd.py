import torch
import torchvision
import torch.utils.data as data
import argparse
import numpy as np
from src.model.base.mlp import MLP
from src.model.cnn import LeNet
from src.model.base.classifier import Classifier
from src.model.makers import make_bayesian_classifier_from_mlps, make_bayesian_classifier_from_lenets
from tqdm import tqdm
import pprint
from copy import deepcopy
import wandb
import math
import os


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--net_type', type=str, default='lenet',
                        help='mlp or lenet')
    parser.add_argument('--dataset_path', type=str, default='/tmp/datasets')
    parser.add_argument('--hidden_layer_sizes', type=list, nargs='+', default=[600],
                        help='affects mlp only')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--learning_rate_prior_and_posterior_mean', type=float, default=0.03)
    parser.add_argument('--learning_rate_posterior_variance', type=float, default=0.003)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--prior_variance_init', type=float, default=0.0003)
    parser.add_argument('--oracle_prior_variance', type=int, default=0)
    parser.add_argument('--prob_threshold', type=float, default=1e-4)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epoch_posterior_mean', type=int, default=256)
    parser.add_argument('--n_epoch_posterior_variance', type=int, default=256)
    parser.add_argument('--n_epoch_prior_mean', type=int, default=256)
    parser.add_argument('--prior_mean_patience', type=int, default=3)
    parser.add_argument('--posterior_mean_stopping_error_train', type=float, default=0.02)

    return parser.parse_args()


def train_classifier_epoch(
        classifier,
        optimizer,
        train_loader,
        train_set_size,
        train_loader_eval,
        test_loader,
        config
):
    training = classifier.training

    classifier.train()
    for x, y in tqdm(train_loader, total=train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)
        probs = classifier(x)
        cross_entropy = classifier.cross_entropy(probs, y).mean()
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


def train_bayesian_classifier_epoch(
        bayesian_classifier,
        optimizer,
        train_loader,
        train_set_size,
        train_loader_eval,
        test_loader,
        config
):
    training = bayesian_classifier.training
    bayesian_classifier.train()

    for x, y in tqdm(train_loader, total=train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)

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

    error_train, surrogate_train = bayesian_classifier.evaluate_on_loader(train_loader_eval)
    error_test, surrogate_test = bayesian_classifier.evaluate_on_loader(test_loader)
    with torch.no_grad():
        kl = bayesian_classifier.kl()
        error_bound = bayesian_classifier.inverted_kl_bound(
            risk=error_train,
            kl=kl,
            dataset_size=train_set_size,
            delta=config.delta
        )
        surrogate_bound = bayesian_classifier.inverted_kl_bound(
            risk=surrogate_train,
            kl=kl,
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
        'kl_normalized': kl.item() / train_set_size
    }

    bayesian_classifier.train(mode=training)

    return log


def l2_between_mlps(mlp1, mlp2):
    mlp1_vector = torch.cat([p.view(-1) for p in mlp1.parameters() if p.requires_grad])
    mlp2_vector = torch.cat([p.view(-1) for p in mlp2.parameters() if p.requires_grad])
    return (mlp1_vector - mlp2_vector).norm()


def main(args):

    pp = pprint.PrettyPrinter()
    wandb.init(project="role-of-data",
               dir='/tmp/output/role_of_data/data_dependent_prior_sgd/debug',
               tags=['debug'])
    config = wandb.config
    config.update(args)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    if config.dataset == 'mnist':
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
    elif config.dataset == 'fashion_mnist':
        train_set = torchvision.datasets.FashionMNIST(
            root=config.dataset_path,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        test_set = torchvision.datasets.FashionMNIST(
            root=config.dataset_path,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )

    else:
        raise ValueError

    train_set_size_all = len(train_set)
    train_set_size_alpha = int(config.alpha * train_set_size_all)
    train_set_size_not_alpha = train_set_size_all - train_set_size_alpha

    train_set_alpha, train_set_not_alpha = data.random_split(
        train_set, [train_set_size_alpha, train_set_size_not_alpha]
    )

    # S, the entire training set
    train_loader_all = data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
    )

    # S but with large batch size; for use in eval (no gradients)
    train_loader_eval_all = data.DataLoader(
        dataset=train_set,
        batch_size=len(train_set)//10,
        num_workers=2,
    )

    if config.alpha != 0:
        # S_alpha
        train_loader_alpha = data.DataLoader(
            dataset=train_set_alpha,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=2
        )
        train_loader_eval_alpha = data.DataLoader(
            dataset=train_set_alpha,
            batch_size=train_set_size_alpha // 10,
            num_workers=2
        )

    # S \ S_alpha
    train_loader_not_alpha = data.DataLoader(
        dataset=train_set_not_alpha,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2
    )
    train_loader_eval_not_alpha = data.DataLoader(
        dataset=train_set_not_alpha,
        batch_size=train_set_size_not_alpha // 10,
        num_workers=2
    )

    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=len(test_set),
        num_workers=2
    )

    if config.net_type == 'mlp':
        net_posterior_mean = MLP(
            n_input=784,
            n_output=10,
            hidden_layer_sizes=config.hidden_layer_sizes
        )
    elif config.net_type == 'lenet':
        net_posterior_mean = LeNet(
            input_shape=[1, 28, 28],
            n_output=10
        )
    else:
        raise ValueError
    classifier_posterior_mean = Classifier(
        net=net_posterior_mean
    )
    classifier_posterior_mean = classifier_posterior_mean.to(device)

    optimizer_posterior_mean = torch.optim.SGD(
        classifier_posterior_mean.parameters(),
        lr=config.learning_rate_prior_and_posterior_mean,
        momentum=config.momentum
    )
    classifier_posterior_mean_init = deepcopy(classifier_posterior_mean).to(device)

    # train for one epoch on S_alpha; coupling
    if config.alpha != 0:
        print('optimizing posterior and prior means for one epoch of S_alpha')
        log = train_classifier_epoch(
            classifier=classifier_posterior_mean,
            optimizer=optimizer_posterior_mean,
            train_loader=train_loader_alpha,
            train_set_size=train_set_size_alpha,
            train_loader_eval=train_loader_eval_alpha,
            test_loader=test_loader,
            config=config
        )
        pp.pprint(log)

    # coupling
    classifier_prior_mean = deepcopy(classifier_posterior_mean).to(device)

    # for the posterior mean, finish off the epoch of S with S \ S_alpha
    print('optimizing posterior mean for one epoch of S \\ S_alpha')
    log = train_classifier_epoch(
        classifier=classifier_posterior_mean,
        optimizer=optimizer_posterior_mean,
        train_loader=train_loader_not_alpha,
        train_set_size=train_set_size_not_alpha,
        train_loader_eval=train_loader_eval_not_alpha,
        test_loader=test_loader,
        config=config
    )
    pp.pprint(log)

    # finish optimizing posterior mean on S
    print(f'optimizing posterior mean to error_train {config.posterior_mean_stopping_error_train}')
    for i_epoch in tqdm(range(1, config.n_epoch_posterior_mean)):
        log = train_classifier_epoch(
            classifier=classifier_posterior_mean,
            optimizer=optimizer_posterior_mean,
            train_loader=train_loader_all,
            train_set_size=train_set_size_all,
            train_loader_eval=train_loader_eval_all,
            test_loader=test_loader,
            config=config
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
        lr=config.learning_rate_prior_and_posterior_mean,
        momentum=config.momentum
    )
    l2_closest = float('inf')
    classifier_prior_mean_closest = deepcopy(classifier_prior_mean)
    i_epoch_closest = 0
    if config.alpha != 0:
        for i_epoch in tqdm(range(1, config.n_epoch_prior_mean)):
            log = train_classifier_epoch(
                classifier=classifier_prior_mean,
                optimizer=optimizer_prior_mean,
                train_loader=train_loader_alpha,
                train_set_size=train_set_size_alpha,
                train_loader_eval=train_loader_eval_alpha,
                test_loader=test_loader,
                config=config
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

    if config.net_type == 'mlp':
        bayesian_classifier = make_bayesian_classifier_from_mlps(
            mlp_posterior_mean_init=classifier_posterior_mean.net,
            mlp_prior_mean=classifier_prior_mean_closest.net,
            prior_stddev=math.sqrt(config.prior_variance_init),
            optimize_prior_mean=False,
            optimize_prior_rho=False,
            optimize_posterior_mean=False,
            optimize_posterior_rho=True,
            prob_threshold=config.prob_threshold,
            normalize_surrogate_by_log_classes=True,
            oracle_prior_variance=config.oracle_prior_variance
        )
    elif config.net_type == 'lenet':
        bayesian_classifier = make_bayesian_classifier_from_lenets(
            net_posterior_mean_init=classifier_posterior_mean.net,
            net_prior_mean=classifier_prior_mean_closest.net,
            prior_stddev=math.sqrt(config.prior_variance_init),
            optimize_prior_mean=False,
            optimize_prior_rho=False,
            optimize_posterior_mean=False,
            optimize_posterior_rho=True,
            prob_threshold=config.prob_threshold,
            normalize_surrogate_by_log_classes=True,
            oracle_prior_variance=config.oracle_prior_variance
        )
    else:
        raise ValueError

    bayesian_classifier = bayesian_classifier.to(device)
    optimizer_posterior_variance = torch.optim.SGD(
        bayesian_classifier.parameters(),
        lr=config.learning_rate_posterior_variance,
        momentum=config.momentum
    )

    error, surrogate = bayesian_classifier.evaluate_on_loader(train_loader_all)
    pp.pprint('bayesian classifier evaluation')
    pp.pprint({'error_train': error, 'surrogate_train': surrogate})

    pp.pprint('bayesian classifier bound optimization')
    for _ in tqdm(range(config.n_epoch_posterior_variance)):
        log = train_bayesian_classifier_epoch(
            bayesian_classifier=bayesian_classifier,
            optimizer=optimizer_posterior_variance,
            train_loader=train_loader_not_alpha,
            train_set_size=train_set_size_not_alpha,
            train_loader_eval=train_loader_eval_not_alpha,
            test_loader=test_loader,
            config=config
        )

        wandb.log(log)
        pp.pprint(log)

    torch.save(bayesian_classifier.state_dict(), os.path.join(wandb.run.dir, 'model.pt'))


if __name__ == '__main__':
    args = parse_args()
    main(args)
