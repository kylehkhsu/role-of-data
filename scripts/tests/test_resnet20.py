import torch
import torchvision
import torchvision.transforms as transforms
import torch.utils.data as data
import argparse
import numpy as np
from scripts.data_dependent_prior_sgd import train_classifier_epoch
from src.model.cnn import resnet20
from src.model.base.classifier import Classifier
from tqdm import tqdm
import pprint
import wandb

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='cifar10')
    parser.add_argument('--net_type', type=str, default='resnet20')
    parser.add_argument('--dataset_path', type=str, default='/tmp/datasets')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--n_epoch', type=int, default=256)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--data_augmentation_and_normalization', type=int, default=0)

    return parser.parse_args()


def main(args):
    pp = pprint.PrettyPrinter()
    wandb.init(project="role-of-data",
               dir='/tmp/output/role_of_data/data_dependent_prior/',
               tags=['debug'])
    config = wandb.config
    config.update(args)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if config.data_augmentation_and_normalization:
        train_set = torchvision.datasets.CIFAR10(
            root=config.dataset_path,
            train=True,
            transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
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
        train_set = torchvision.datasets.CIFAR10(
            root=config.dataset_path,
            train=True,
            transform=transforms.ToTensor(),
            download=True
        )

        test_set = torchvision.datasets.CIFAR10(
            root=config.dataset_path,
            train=False,
            transform=transforms.ToTensor(),
            download=True
        )
    train_set_size = len(train_set)

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4
    )
    train_loader_eval = data.DataLoader(
        dataset=train_set,
        batch_size=train_set_size // 10,
        shuffle=True,
        num_workers=4
    )
    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=len(test_set),
        num_workers=4
    )

    net = resnet20()

    classifier = Classifier(net=net).to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum,
        weight_decay=config.weight_decay
    )

    for i_epoch in tqdm(range(config.n_epoch)):
        log = train_classifier_epoch(
            classifier=classifier,
            optimizer=optimizer,
            train_loader=train_loader,
            train_set_size=train_set_size,
            train_loader_eval=train_loader_eval,
            test_loader=test_loader,
            config=config
        )
        wandb.log(log)
        pp.pprint(log)


if __name__ == '__main__':
    args = parse_args()
    main(args)
