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
            nn.Conv2d(1, 3, 5),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(3),
            nn.Flatten()
        )

        n_channels = self.feature_extractor(torch.empty(1, 1, CONFIG['IMAGE_RESIZE'], CONFIG['IMAGE_RESIZE'])).size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 84),
            nn.ReLU(),
            nn.Dropout(0.25),
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
            nn.Conv2d(1, 3, 5),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(3),
            nn.Flatten()
        )

        n_channels = self.feature_extractor(torch.empty(1, 1, 128, 128)).size(-1)

        self.classifier = nn.Sequential(
            nn.Linear(n_channels, 84),
            nn.ReLU(),
            nn.Linear(84, 2))

    def forward(self, x):
        features = self.feature_extractor(x)
        out = self.classifier(features)
        return out


class LeNet5(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(6),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out


class LeNet5NoRegNoBN(nn.Module):
    def __init__(self, num_classes):
        super(LeNet5NoRegNoBN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
            nn.MaxPool2d(kernel_size = 2, stride = 2))

        self.fc = nn.Linear(400, 120)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(120, 84)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.relu(out)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        return out



def save_model(model, name: str):
    CONFIG = helpers.get_config()
    PATH = CONFIG['MODELS_FOLDER'] + name
    torch.save(model.state_dict(), PATH)
    return None
