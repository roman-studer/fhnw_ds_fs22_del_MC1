import pandas as pd
import wandb
import torch
import numpy as np
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import optuna

import sys

sys.path.insert(0, '../')
from helpers import model_pipeline, helpers, data_loader


def nn_train(model, device, train_dataloader, optimizer, criterion, epoch, scheduler, steps_per_epoch=20):
    model.train()

    train_loss = 0
    train_total = 0
    train_correct = 0

    for batch_idx, (data, target) in enumerate(train_dataloader, start=0):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()

        output = model(data)

        loss = criterion(output, target)
        train_loss += loss.item()

        scores, predictions = torch.max(output.data, 1)
        train_total += target.size(0)
        train_correct += int(sum(predictions == target))

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()
    scheduler.step()

    acc = round((train_correct / train_total) * 100, 2)
    print("Epoch [{}], Loss: {}, Accuracy: {}".format(epoch, train_loss / train_total, acc), end="")
    wandb.log({"Train Loss": train_loss / train_total, "Train Accuracy": acc, "Epoch": epoch})

    return None


def nn_test(model, device, test_dataloader, criterion, classes, return_prediction=False):
    model.eval()

    # test model
    test_loss = 0
    test_total = 0
    test_correct = 0

    y_true = []
    y_pred = []
    y_proba = None

    with torch.no_grad():
        for data in test_dataloader:
            inputs, labels = data[0].to(device), data[1].to(device)

            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            if test_loss is None:
                print(f'[WARN] Test loss is nan: {test_loss}')

            scores, predictions = torch.max(outputs.data, 1)
            test_total += labels.size(0)
            test_correct += int(sum(predictions == labels))

            if y_proba is None:
                y_proba = outputs.cpu()
            else:
                y_proba = np.vstack((y_proba, outputs.cpu()))

            for i in labels.tolist():
                y_true.append(i)
            for j in predictions.tolist():
                y_pred.append(j)

    if return_prediction:
        return y_true, y_pred, y_proba

    acc = round((test_correct / test_total) * 100, 2)
    print(" Test_loss: {}, Test_accuracy: {}".format(test_loss / test_total, acc))
    wandb.log({"Test Loss": test_loss / test_total, "Test Accuracy": acc})

    return acc


CONFIG = helpers.get_config()

DEVICE = CONFIG['DEVICE']
LEARNING_RATE = CONFIG['LEARNING_RATE']
BATCH_SIZE = CONFIG['BATCH_SIZE']
EPOCHS = CONFIG['EPOCHS']
TRANSFORMER = CONFIG['TRANSFORMER']
TRAIN_SET = CONFIG['DATA_DIR_TRAINSET']
TEST_SET = CONFIG['DATA_DIR_TESTSET']
RAW_DATA = CONFIG['DATA_DIR_RESIZED']
PLOT_DIR = CONFIG['PLOT_DIR_BINARY']
NET = CONFIG['NET']
CRITERION = CONFIG['CRITERION']
OPTIMIZER = CONFIG['OPTIMIZER']
TASK = CONFIG['TASK']
SAFE_MODEL = CONFIG['SAVE_MODEL']
SCHEDULER = CONFIG['SCHEDULER']
DROPOUT_RATE = CONFIG['DROPOUT_RATE']


def objective(trial):
    BATCH_SIZE = trial.suggest_int('batch size', 2, 16, 2)
    LEARNING_RATE = trial.suggest_float('learning rate', 0.00001, 0.05)

    wandb.init(project="del_mc1", entity="ryzash01", dir='../models/',
               name=f'Net: {NET} Transf: {TRANSFORMER} Epochs: {EPOCHS}',
               settings=wandb.Settings(_disable_stats=True))

    wandb.define_metric("acc", summary="max")
    wandb.define_metric("loss", summary="min")

    wandb.config = {
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "transformer": TRANSFORMER,
        "net": NET,
        "criterion": CRITERION,
        "optimizer": OPTIMIZER,
        "image_size": CONFIG['IMAGE_RESIZE'],
        "random_state": CONFIG['RANDOM_STATE'],
        "scheduler": SCHEDULER,
        "dropout_rate": DROPOUT_RATE
    }

    class_names = ['Atelectasis', 'Effusion']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using {device} for training')
    if device == 'cuda':
        torch.cuda.empty_cache()

    if NET == 'Cxr8Net':
        net = model_pipeline.Cxr8Net().to(device)
    elif NET == 'Cxr8NetNoRegNoBN':
        net = model_pipeline.Cxr8NetNoRegNoBN().to(device)
    elif NET == 'LeNet5':
        net = model_pipeline.LeNet5(num_classes=2).to(device)
    elif NET == 'LeNet5NoRegNoBN':
        net = model_pipeline.LeNet5NoRegNoBN(num_classes=2).to(device)

    print(f'[INFO] Net: {NET}')
    print(f'[INFO] Params: LR = {LEARNING_RATE}, DO = {DROPOUT_RATE}, BS = {BATCH_SIZE}, EP = {EPOCHS}')

    if TRANSFORMER == 'Cxr8' or TRANSFORMER is None:
        transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=0, std=1)
        ])

    elif TRANSFORMER == 'LeNet5':
        transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=0, std=1),
            transforms.Resize((CONFIG['IMAGE_RESIZE'], CONFIG['IMAGE_RESIZE']))
        ])

    print(f'[INFO] Transformer: {TRANSFORMER}')

    test_transform, train_transform = transformer, transformer

    train_data = data_loader.Cxr8ImageDatasetLoader(annotations_file=TRAIN_SET, img_dir=RAW_DATA,
                                                    transform=train_transform)
    test_data = data_loader.Cxr8ImageDatasetLoader(annotations_file=TEST_SET, img_dir=RAW_DATA,
                                                   transform=test_transform)

    train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=True)

    if CRITERION == 'Cross Entropy' or CRITERION is None:
        criterion = nn.CrossEntropyLoss()

    print(f'[INFO] Criterion: {CRITERION}')

    if OPTIMIZER == 'Adam' or OPTIMIZER is None:
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE, weight_decay=0.0001)
    elif OPTIMIZER == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

    if SCHEDULER == "ExponentialLR" or SCHEDULER is None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    print(f'[INFO] Optimizer: {OPTIMIZER}, learning rate: {LEARNING_RATE}')

    wandb.watch(net, log="parameters")

    print('\n[INFO] Started Training')

    best_test_acc = 0

    for epoch in range(EPOCHS):
        nn_train(net, device, train_dataloader, optimizer, criterion, epoch, scheduler)
        test_acc = nn_test(net, device, test_dataloader, criterion, class_names)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc

    y_true, y_pred, y_proba = nn_test(net, device, test_dataloader, criterion, class_names, return_prediction=True)
    wandb.log(
        {"roc": wandb.plot.roc_curve(np.array(y_true), np.array(y_proba), labels=class_names, classes_to_plot=None),
         "learning_rate": LEARNING_RATE,
         "epochs": EPOCHS,
         "batch_size": BATCH_SIZE,
         "transformer": TRANSFORMER,
         "net": NET,
         "criterion": CRITERION,
         "optimizer": OPTIMIZER})

    print("[INFO] Finished Training\n")
    wandb.finish()

    if device == 'cuda':
        torch.cuda.empty_cache()

    return best_test_acc


study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=10)
