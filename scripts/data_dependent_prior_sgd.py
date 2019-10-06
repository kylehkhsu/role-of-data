import sys

sys.path.insert(1, './')
import torch
import torch.utils.data as data
import argparse
import numpy as np
from tqdm import tqdm
import pprint
from copy import deepcopy
import wandb
import math
import os
from src.util import device, get_dataset, train_classifier_epoch, train_bayesian_classifier_epoch, make_classifier, \
    make_bayesian_classifier


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='mnist')
    parser.add_argument('--net_type', type=str, default='mlp',
                        help='mlp or lenet')
    parser.add_argument('--dataset_path', type=str, default='/h/kylehsu/datasets')
    parser.add_argument('--hidden_layer_sizes', type=list, nargs='+', default=[600],
                        help='affects mlp only and not lenet')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--alpha', type=float, default=0.7)
    parser.add_argument('--learning_rate_prior_and_posterior_mean', type=float, default=3e-3)
    parser.add_argument('--learning_rate_posterior_variance', type=float, default=3e-5)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--prior_variance_init', type=float, default=0.003)
    parser.add_argument('--oracle_prior_variance', type=int, default=0)
    parser.add_argument('--prob_threshold', type=float, default=1e-4)
    parser.add_argument('--delta', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epoch_posterior_mean', type=int, default=256)
    parser.add_argument('--n_epoch_posterior_variance', type=int, default=256)
    parser.add_argument('--n_epoch_prior_mean', type=int, default=256)
    parser.add_argument('--prior_mean_patience', type=int, default=5)
    parser.add_argument('--posterior_mean_stopping_error_train', type=float, default=0.02)

    return parser.parse_args()


def l2_between_mlps(mlp1, mlp2):
    mlp1_vector = torch.cat([p.view(-1) for p in mlp1.parameters() if p.requires_grad])
    mlp2_vector = torch.cat([p.view(-1) for p in mlp2.parameters() if p.requires_grad])
    return (mlp1_vector - mlp2_vector).norm()


def main(args):
    pp = pprint.PrettyPrinter()
    wandb.init(project="pacbayes_opt",
               dir='/scratch/hdd001/home/kylehsu/output/pacbayes_opt/data_dependent_prior_sgd/debug',
               tags=['debug'])
    config = wandb.config
    config.update(args)
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    train_set, test_set = get_dataset(config)

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
        batch_size=len(train_set) // 10,
        num_workers=2,
    )

    train_loader_alpha, train_loader_eval_alpha = None, None
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

    classifier_posterior_mean = make_classifier(config).to(device)

    optimizer_posterior_mean = torch.optim.SGD(
        classifier_posterior_mean.parameters(),
        lr=config.learning_rate_prior_and_posterior_mean,
        momentum=config.momentum
    )
    classifier_posterior_mean_init = deepcopy(classifier_posterior_mean).to(device)

    # train for one epoch on S_alpha; coupling
    if config.alpha != 0:
        assert train_loader_alpha is not None and train_loader_eval_alpha is not None
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
    optimizer_prior_mean = torch.optim.SGD(
        classifier_prior_mean.parameters(),
        lr=config.learning_rate_prior_and_posterior_mean,
        momentum=config.momentum
    )
    optimizer_prior_mean.load_state_dict(optimizer_posterior_mean.state_dict())

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

    print('l2 distance between trained posterior mean and initial posterior mean:',
          f'{l2_between_mlps(classifier_posterior_mean.net, classifier_posterior_mean_init.net)}')
    print('l2 distance between trained posterior mean and initial prior mean:',
          f'{l2_between_mlps(classifier_posterior_mean.net, classifier_prior_mean.net)}')
    print(f'l2 distance between initial posterior mean and initial prior mean:',
          f'{l2_between_mlps(classifier_posterior_mean_init.net, classifier_prior_mean.net)}')

    l2_closest = float('inf')
    classifier_prior_mean_closest = deepcopy(classifier_prior_mean)
    i_epoch_closest = 0
    if config.alpha != 0:
        assert train_loader_alpha is not None and train_loader_eval_alpha is not None

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

    bayesian_classifier = make_bayesian_classifier(
        config=config,
        classifier_prior_mean=classifier_prior_mean_closest,
        classifier_posterior_mean=classifier_posterior_mean,
        optimize_prior_mean=False,
        optimize_prior_rho=False,
        optimize_posterior_mean=False,
        optimize_posterior_rho=True,
        oracle_prior_variance=config.oracle_prior_variance
    )
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
