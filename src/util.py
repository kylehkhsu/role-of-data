import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
import math
from src.model.base.mlp import MLP
from src.model.base.lenet import LeNet
from src.model.base.resnet import resnet20
from src.model.base.classifier import Classifier
from src.model.bayesian.bayesian_mlp import BayesianMLP
from src.model.bayesian.bayesian_lenet import BayesianLeNet
from src.model.bayesian.bayesian_resnet import BayesianResNet
from src.model.bayesian.bayesian_classifier import BayesianClassifier

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def get_dataset(config):
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
    elif config.dataset == 'cifar10':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_set = torchvision.datasets.CIFAR10(
            root=config.dataset_path,
            train=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=True
        )
        test_set = torchvision.datasets.CIFAR10(
            root=config.dataset_path,
            train=False,
            transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ]),
            download=True
        )
    else:
        raise ValueError

    return train_set, test_set


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


def make_classifier(config):
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
    elif config.net_type == 'resnet20':
        net = resnet20()
    else:
        raise ValueError

    classifier = Classifier(
        net=net
    )

    return classifier


def make_bayesian_classifier(
        config,
        classifier_prior_mean,
        classifier_posterior_mean,
        optimize_prior_mean,
        optimize_prior_rho,
        optimize_posterior_mean,
        optimize_posterior_rho,
        oracle_prior_variance
):
    if config.net_type == 'mlp':
        BayesianNet = BayesianMLP
    elif config.net_type == 'lenet':
        BayesianNet = BayesianLeNet
    elif config.net_type == 'resnet20':
        BayesianNet = BayesianResNet
    else:
        raise ValueError
    bayesian_net = BayesianNet(
        classifier_prior_mean.net,
        classifier_posterior_mean.net,
        prior_stddev=math.sqrt(config.prior_variance_init),
        optimize_prior_mean=optimize_prior_mean,
        optimize_prior_rho=optimize_prior_rho,
        optimize_posterior_mean=optimize_posterior_mean,
        optimize_posterior_rho=optimize_posterior_rho,
    )

    bayesian_classifier = BayesianClassifier(
        bayesian_net=bayesian_net,
        prob_threshold=config.prob_threshold,
        normalize_surrogate_by_log_classes=True,
        oracle_prior_variance=oracle_prior_variance
    )

    return bayesian_classifier
