import torch.nn as nn
import torch.nn.functional as F
import torch
import sys

sys.path.insert(0, '../helpers/')
from helpers import helpers


class Cxr8Net(nn.Module):
    def __init__(self):
        super(Cxr8Net, self).__init__()

        CONFIG = helpers.get_config()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(6, 6),
            nn.BatchNorm2d(3),
            nn.Flatten()
        )

        n_channels = self.feature_extractor(torch.empty(CONFIG['BATCH_SIZE'], 1, CONFIG['IMAGE_RESIZE'], CONFIG['IMAGE_RESIZE'])).size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 84),
            nn.ReLU(),
            nn.Dropout(CONFIG['DROPOUT_RATE']),
            nn.Linear(84, 2))

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out


class Cxr8NetNoRegNoBN(nn.Module):
    def __init__(self):
        super(Cxr8NetNoRegNoBN, self).__init__()

        CONFIG = helpers.get_config()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=5, stride=3, padding=0),
            nn.ReLU(),
            nn.MaxPool2d(6, 6),
            nn.Flatten()
        )

        n_channels = self.feature_extractor(torch.empty(CONFIG['BATCH_SIZE'], 1, CONFIG['IMAGE_RESIZE'], CONFIG['IMAGE_RESIZE'])).size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 84),
            nn.ReLU(),
            nn.Linear(84, 2))

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out


class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        CONFIG = helpers.get_config()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(6),
            nn.ReLU())

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Flatten())


        n_channels = self.layer2(self.layer1(torch.empty(CONFIG['BATCH_SIZE'], 1, CONFIG['IMAGE_RESIZE'], CONFIG['IMAGE_RESIZE']))).size(-1)

        self.dense = nn.Sequential(
            nn.Linear(n_channels, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2))


    def forward(self, x):
        layer1 = self.layer1(x)
        layer2 = self.layer2(layer1)
        out = self.dense(layer2)
        return out


class LeNet5NoRegNoBN(nn.Module):
    def __init__(self):
        super(LeNet5NoRegNoBN, self).__init__()

        CONFIG = helpers.get_config()

        self.cnns = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.ReLU(),
            nn.Flatten())

        n_channels = self.cnns(torch.empty(CONFIG['BATCH_SIZE'], 1, CONFIG['IMAGE_RESIZE'], CONFIG['IMAGE_RESIZE'])).size(-1)

        self.dense = nn.Sequential(
            nn.Linear(n_channels, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2))

    def forward(self, x):
        features = self.cnns(x)
        out = self.dense(features)
        return out


class ComplexCNN(nn.Module):
    def __init__(self):
        super(ComplexCNN, self).__init__()

        CONFIG = helpers.get_config()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=12, stride=3, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(6),
            nn.LeakyReLU(),
            nn.Conv2d(6, 10, kernel_size=6, stride=2, padding=0),
            nn.AvgPool2d(kernel_size=3, stride=2),
            nn.BatchNorm2d(10),
            nn.LeakyReLU(),
            nn.Conv2d(10, 16, kernel_size=4, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        n_channels = self.feature_extractor(torch.empty(CONFIG['BATCH_SIZE'], 1, CONFIG['IMAGE_RESIZE'], CONFIG['IMAGE_RESIZE'])).size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 512),
            nn.LeakyReLU(),
            nn.Dropout(CONFIG['DROPOUT_RATE']),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout(CONFIG['DROPOUT_RATE']),
            nn.Linear(256, 84),
            nn.LeakyReLU(),
            nn.Dropout(CONFIG['DROPOUT_RATE']),
            nn.Linear(84, 2))

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out


class ComplexCNNNoRegNoBN(nn.Module):
    def __init__(self):
        super(ComplexCNNNoRegNoBN, self).__init__()

        CONFIG = helpers.get_config()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=12, stride=3, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(6, 10, kernel_size=6, stride=2, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Conv2d(10, 5, kernel_size=4, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.LeakyReLU(),
            nn.Flatten()
        )

        n_channels = self.feature_extractor(torch.empty(CONFIG['BATCH_SIZE'], 1, CONFIG['IMAGE_RESIZE'], CONFIG['IMAGE_RESIZE'])).size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 512),
            nn.LeakyReLU(),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 84),
            nn.LeakyReLU(),
            nn.Linear(84, 2))

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out


def save_model(model, name: str):
    CONFIG = helpers.get_config()
    PATH = CONFIG['MODELS_FOLDER'] + name
    torch.save(model.state_dict(), PATH)
    return None
