import torch
import torchvision
import torch.utils.data as data
import argparse
import numpy as np
import ipdb
from src.model.base.classifier import Classifier
from src.model.base.mlp import MLP
from tqdm import tqdm
import pprint

pp = pprint.PrettyPrinter()

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_path', type=str, default='/h/kylehsu/datasets')
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--learning_rate', type=float, default=1e-2)
parser.add_argument('--momentum', type=float, default=0.95)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--training_epochs', type=int, default=256)

config = parser.parse_args()

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
    drop_last=True,
    num_workers=2,
)

test_loader = data.DataLoader(
    dataset=test_set,
    batch_size=len(test_set),
    num_workers=2
)

mlp = MLP(
    n_input=784,
    n_output=10,
    hidden_layer_sizes=[600, 600]
)

classifier = Classifier(
    net=mlp
)
classifier = classifier.to(device)
optimizer = torch.optim.SGD(
    classifier.parameters(),
    lr=config.learning_rate,
    momentum=config.momentum
)
ipdb.set_trace()


def evaluate_classifier(classifier):
    classifier.eval()
    with torch.no_grad():
        for x, y in test_loader:
            assert y.shape[0] == len(test_set)
            x, y = x.to(device), y.to(device)
            probs = classifier(x)
            correct, total = classifier.evaluate(probs, y)
            error = 1 - correct / total
    return error


for i_epoch in tqdm(range(config.training_epochs)):
    classifier.train()
    losses = []
    corrects, totals = 0.0, 0.0
    for x, y in tqdm(train_loader, total=train_set_size // config.batch_size):
        x, y = x.to(device), y.to(device)
        probs = classifier(x)
        loss = classifier.loss(probs, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            correct, total = classifier.evaluate(probs, y)
        losses.append(loss.item())
        totals += total.item()
        corrects += correct.item()
    error_train = 1 - corrects / totals
    error_test = evaluate_classifier(classifier)

    log = {
        'epoch': i_epoch,
        'loss_train': np.mean(losses),
        'error_train': error_train,
        'error_test': error_test.item()
    }
    pp.pprint(log)
    if error_train <= 0.01:
        break
