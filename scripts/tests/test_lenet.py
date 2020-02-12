import torch
import torchvision
import torch.utils.data as data
import argparse
import numpy as np
from src.model.base.classifier import Classifier
from src.model.cnn import LeNet
from scripts.data_dependent_prior_sgd import train_classifier_epoch

from tqdm import tqdm
import pprint

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, default='/tmp/datasets')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--learning_rate', type=float, default=1e-2)
    parser.add_argument('--momentum', type=float, default=0.95)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--n_epoch', type=int, default=256)

    return parser.parse_args()


def main(args):
    config = args
    pp = pprint.PrettyPrinter()

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

    train_loader = data.DataLoader(
        dataset=train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=2,
    )
    train_loader_eval = data.DataLoader(
        dataset=train_set,
        batch_size=train_set_size // 10,
        num_workers=2,
    )

    test_loader = data.DataLoader(
        dataset=test_set,
        batch_size=len(test_set),
        num_workers=2
    )

    lenet = LeNet(
        input_shape=[1, 28, 28],
        n_output=10
    )

    classifier = Classifier(
        net=lenet
    )
    classifier = classifier.to(device)
    optimizer = torch.optim.SGD(
        classifier.parameters(),
        lr=config.learning_rate,
        momentum=config.momentum
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

        log.update({
            'epoch': i_epoch,
        })
        pp.pprint(log)


if __name__ == '__main__':
    args = parse_args()
    main(args)
