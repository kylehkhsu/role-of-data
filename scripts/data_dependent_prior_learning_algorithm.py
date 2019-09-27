import torch
import torchvision
import torch.utils.data as data
import argparse
import numpy as np
import ipdb
from scripts.data_dependent_prior_sgd import train_classifier_epoch, train_bayesian_classifier_epoch
from src.model.mlp import MLP
from src.model.cnn import LeNet
from src.model.classifier import Classifier
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
    parser.add_argument('--dataset_path', type=str, default='/h/kylehsu/datasets')
    parser.add_argument('--alpha', type=float, default=0.1)
    parser.add_argument('--learning_rate_prior', type=float, default=1e-2)
    parser.add_argument('--learning_rate_posterior', type=float, default=0.00001)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_layer_sizes', type=list, nargs='+', default=[600] * 3)
    parser.add_argument('--n_epoch_prior', type=int, default=100)
    parser.add_argument('--n_epoch_posterior', type=int, default=256)
    parser.add_argument('--prior_variance_init', type=float, default=1e-7)
    parser.add_argument('--prob_threshold', type=float, default=1e-4)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--net_type', type=str, default='mlp',
                        help='mlp or lenet')
    return parser.parse_args()


def main(args):
    pp = pprint.PrettyPrinter()
    wandb.init(project="pacbayes_opt",
               dir='/scratch/hdd001/home/kylehsu/output/pacbayes_opt/data_dependent_prior/',
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
    else:
        raise ValueError

    train_set_size = len(train_set)
    train_set_size_alpha = int(config.alpha * train_set_size)
    train_set_size_not_alpha = train_set_size - train_set_size_alpha

    train_set_alpha, train_set_not_alpha = data.random_split(
        train_set, [train_set_size_alpha, train_set_size_not_alpha]
    )
    print(
        f'alpha: {config.alpha}',
        f'train_set_size_alpha: {train_set_size_alpha}',
        f'train_set_size_not_alpha: {train_set_size_not_alpha}'
    )

    if config.alpha != 0:
        train_loader_alpha = data.DataLoader(
            dataset=train_set_alpha,
            batch_size=config.batch_size,
            shuffle=True,
            drop_last=True,
            num_workers=2
        )
        train_loader_eval_alpha = data.DataLoader(
            dataset=train_set_alpha,
            batch_size=train_set_size_alpha // 10,
            num_workers=2
        )
    train_loader_not_alpha = data.DataLoader(
        dataset=train_set_not_alpha,
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=True,
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
        net = MLP(
            n_input=784,
            n_output=10,
            hidden_layer_sizes=config.hidden_layer_sizes
        )
    elif config.net_type == 'lenet':
        net = LeNet(
            input_shape=[1, 28, 28],
            n_output=10
        )
    else:
        raise ValueError

    classifier = Classifier(
        net=net
    )
    classifier = classifier.to(device)

    optimizer_prior = torch.optim.SGD(
        classifier.parameters(),
        lr=config.learning_rate_prior,
        momentum=config.momentum
    )

    classifier_posterior_mean_init = None

    if config.alpha != 0:
        for i_epoch in tqdm(range(config.n_epoch_prior)):
            log = train_classifier_epoch(
                classifier=classifier,
                optimizer=optimizer_prior,
                train_loader=train_loader_alpha,
                train_set_size=train_set_size_alpha,
                train_loader_eval=train_loader_eval_alpha,
                test_loader=test_loader,
                config=config
            )

            if i_epoch == 0:
                classifier_posterior_mean_init = deepcopy(classifier).to('cpu')
                print('saved copy of classifier as classifier_posterior_mean_init')

            log.update({'prior_training_epoch': i_epoch})
            pp.pprint(log)

            if log['error_train'] < 1e-5:
                break

    classifier_prior_mean = classifier.to('cpu')
    if classifier_posterior_mean_init is None:
        assert config.alpha == 0 or config.n_epoch_prior == 0
        print('no prior was trained')
        classifier_posterior_mean_init = deepcopy(classifier).to('cpu')

    if config.net_type == 'mlp':
        bayesian_classifier = make_bayesian_classifier_from_mlps(
            mlp_posterior_mean_init=classifier_posterior_mean_init.net,
            mlp_prior_mean=classifier_prior_mean.net,
            prior_stddev=math.sqrt(config.prior_variance_init),
            optimize_prior_mean=False,
            optimize_prior_rho=False,
            optimize_posterior_mean=True,
            optimize_posterior_rho=True,
            probability_threshold=config.prob_threshold,
            normalize_surrogate_by_log_classes=True
        )
    elif config.net_type == 'lenet':
        bayesian_classifier = make_bayesian_classifier_from_lenets(
            net_posterior_mean_init=classifier_posterior_mean_init.net,
            net_prior_mean=classifier_prior_mean.net,
            prior_stddev=math.sqrt(config.prior_variance_init),
            optimize_prior_mean=False,
            optimize_prior_rho=False,
            optimize_posterior_mean=True,
            optimize_posterior_rho=True,
            probability_threshold=config.prob_threshold,
            normalize_surrogate_by_log_classes=True
        )
    else:
        raise ValueError

    bayesian_classifier = bayesian_classifier.to(device)
    optimizer_posterior = torch.optim.SGD(
        bayesian_classifier.parameters(),
        lr=config.learning_rate_posterior,
        momentum=config.momentum
    )

    for _ in tqdm(range(config.n_epoch_posterior)):
        log = train_bayesian_classifier_epoch(
            bayesian_classifier=bayesian_classifier,
            optimizer=optimizer_posterior,
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
