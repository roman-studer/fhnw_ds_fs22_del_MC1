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

        #torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

        optimizer.step()

    scheduler.step()

    acc = round((train_correct / train_total) * 100, 2)
    print("Epoch [{}], Loss: {}, Accuracy: {}".format(epoch, train_loss / train_total, acc), end="")
    wandb.log({"Train Loss": train_loss / train_total, "Train Accuracy": acc, "Epoch": epoch,
               "learning_rate": scheduler.get_last_lr()[0]})

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

global LEARNING_RATE, OPTIMIZER
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
    global LEARNING_RATE, BATCH_SIZE, net, OPTIMIZER
    if LEARNING_PARAMETER == 1:
        BATCH_SIZE = trial.suggest_int('batch size', 2, 32, 2)
        LEARNING_RATE = CONFIG['LEARNING_RATE']

    elif LEARNING_PARAMETER == 2:
        BATCH_SIZE = CONFIG['BATCH_SIZE']
        LEARNING_RATE = trial.suggest_float('learning rate', 0.0001, 0.01)

    elif LEARNING_PARAMETER == 3:
        L2 = trial.suggest_float('l2_reg', 0, 0.3)
        BATCH_SIZE = CONFIG['BATCH_SIZE']
        LEARNING_RATE = CONFIG['LEARNING_RATE']
        OPTIMIZER = 'Adam'

    elif LEARNING_PARAMETER == 0:
        BATCH_SIZE = CONFIG['BATCH_SIZE']
        LEARNING_RATE = CONFIG['LEARNING_RATE']


    wandb.init(project="del_mc1", entity="ryzash01", dir='../models/',
               name=f'Net: {NET} Transf: {TRANSFORMER} Epochs: {EPOCHS}',
               settings=wandb.Settings(_disable_stats=True),
               tags=TAGS)

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

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f'[INFO] Using {device} for training')
    if device == 'cuda:0':
        torch.cuda.empty_cache()

    if NET == 'Cxr8Net':
        net = model_pipeline.Cxr8Net().to(device)
        NCONVOLUTIONS = 1
        IN_CHANNELS = 1
        OUT_CHANNELS = 3
        KERNEL_SIZE = 5
        STRIDE = 1
        PADDING = 0

    elif NET == 'Cxr8NetNoRegNoBN':
        net = model_pipeline.Cxr8NetNoRegNoBN().to(device)
        NCONVOLUTIONS = 1
        IN_CHANNELS = 1
        OUT_CHANNELS = 3
        KERNEL_SIZE = 5
        STRIDE = 1
        PADDING = 0

    elif NET == 'LeNet5':
        net = model_pipeline.LeNet5().to(device)
        NCONVOLUTIONS = 2
        IN_CHANNELS = [1, 6]
        OUT_CHANNELS = [6, 16]
        KERNEL_SIZE = [5, 5]
        STRIDE = [1, 1]
        PADDING = [0, 0]

    elif NET == 'LeNet5NoRegNoBN':
        net = model_pipeline.LeNet5NoRegNoBN().to(device)
        NCONVOLUTIONS = 2
        IN_CHANNELS = [1, 6]
        OUT_CHANNELS = [6, 16]
        KERNEL_SIZE = [5, 5]
        STRIDE = [1, 1]
        PADDING = [0, 0]

    elif NET == 'ComplexCNN':
        net = model_pipeline.ComplexCNN().to(device)
        NCONVOLUTIONS = 3
        IN_CHANNELS = [1, 6, 10]
        OUT_CHANNELS = [6, 10, 5]
        KERNEL_SIZE = [12, 6, 4]
        STRIDE = [3, 2, 1]
        PADDING = [0, 0, 0]

    elif NET == 'ComplexCNNNoRegNoBN':
        net = model_pipeline.ComplexCNNNoRegNoBN().to(device)
        NCONVOLUTIONS = 3
        IN_CHANNELS = [1, 6, 10]
        OUT_CHANNELS = [6, 10, 5]
        KERNEL_SIZE = [12, 6, 4]
        STRIDE = [3, 2, 1]
        PADDING = [0, 0, 0]

    print(f'[INFO] Net: {NET}')
    print(f'[INFO] Params: LR = {LEARNING_RATE}, DO = {DROPOUT_RATE}, BS = {BATCH_SIZE}, EP = {EPOCHS}')

    if TRANSFORMER == 'Cxr8' or TRANSFORMER is None:
        transformer = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
            transforms.Grayscale(num_output_channels=1),
            transforms.Normalize(mean=0, std=1),
            transforms.Resize((CONFIG['IMAGE_RESIZE'], CONFIG['IMAGE_RESIZE']))
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
        optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'AdamW' or OPTIMIZER is None:
        optimizer = optim.AdamW(net.parameters(), lr=LEARNING_RATE)
    elif OPTIMIZER == 'SGD':
        if LEARNING_PARAMETER == 3:
            optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE, weight_decay=L2)
        else:
            optimizer = optim.SGD(net.parameters(), lr=LEARNING_RATE)

    if SCHEDULER == "ExponentialLR" or SCHEDULER is None:
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    print(f'[INFO] Optimizer: {OPTIMIZER}, learning rate: {LEARNING_RATE}')


    print('\n[INFO] Started Training')

    best_test_acc = 0

    for epoch in range(EPOCHS):
        nn_train(net, device, train_dataloader, optimizer, criterion, epoch, scheduler)
        test_acc = nn_test(net, device, test_dataloader, criterion, class_names)

        if test_acc >= best_test_acc:
            best_test_acc = test_acc

        trial.report(test_acc, epoch)

        # Handle pruning based on the test accuracy
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()


    y_true, y_pred, y_proba = nn_test(net, device, test_dataloader, criterion, class_names, return_prediction=True)
    if np.isnan(y_true).any() or np.isnan(y_proba).any():
        wandb.log(
            {"learning_rate": LEARNING_RATE,
             "epochs": EPOCHS,
             "batch_size": BATCH_SIZE,
             "transformer": TRANSFORMER,
             "net": NET,
             "criterion": CRITERION,
             "optimizer": OPTIMIZER,
             "nconvolutions": NCONVOLUTIONS,
             "kernel_size": KERNEL_SIZE,
             "in_channels": IN_CHANNELS,
             "out_channels": OUT_CHANNELS,
             "stride": STRIDE,
             "PADDING": PADDING})
    else:
        wandb.log(
            {"roc": wandb.plot.roc_curve(np.array(y_true), np.array(y_proba), labels=class_names, classes_to_plot=None),
             "learning_rate": LEARNING_RATE,
             "epochs": EPOCHS,
             "batch_size": BATCH_SIZE,
             "transformer": TRANSFORMER,
             "net": NET,
             "criterion": CRITERION,
             "optimizer": OPTIMIZER,
             "nconvolutions": NCONVOLUTIONS,
             "kernel_size": KERNEL_SIZE,
             "in_channels": IN_CHANNELS,
             "out_channels": OUT_CHANNELS,
             "stride": STRIDE,
             "PADDING": PADDING})

    print("[INFO] Finished Training\n")
    wandb.finish()

    if device == 'cuda:o':
        torch.cuda.empty_cache()

    return best_test_acc



global LEARNING_PARAMETER
LEARNING_PARAMETER = 1
TAGS = ['Batch Size', 'No Reg']

# study = optuna.create_study(direction='maximize',
#                             pruner=optuna.pruners.MedianPruner())
# NET = 'LeNet5NoRegNoBN'
# study.optimize(objective, n_trials=3)

LEARNING_PARAMETER = 2
TAGS = ['Learning Rate', 'No Reg']


LEARNING_PARAMETER = 3
TAGS = ['L2 Norm', 'No Reg']


study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.MedianPruner())
NET = 'ComplexCNNNoRegNoBN'
study.optimize(objective, n_trials=1)

study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.MedianPruner())
NET = 'LeNet5NoRegNoBN'
study.optimize(objective, n_trials=3)

LEARNING_PARAMETER = 0
TAGS = ['Dropout', 'Batchnorm']

study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.MedianPruner())
NET = 'Cxr8Net'
study.optimize(objective, n_trials=1)

study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.MedianPruner())
NET = 'ComplexCNN'
study.optimize(objective, n_trials=1)

study = optuna.create_study(direction='maximize',
                            pruner=optuna.pruners.MedianPruner())
NET = 'LeNet5'
study.optimize(objective, n_trials=1)


LEARNING_PARAMETER = 0
TAGS = ['Adam', 'No Reg']
OPTIMIZER = 'Adam'


LEARNING_PARAMETER = 3
TAGS = ['Adam', 'Reg']



