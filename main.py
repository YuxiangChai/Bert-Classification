import torch
import torch.nn as nn
import torch.optim as optim
import time
from transformers import BertForSequenceClassification
from torch.utils.data import random_split, DataLoader

from dataset import DataSet
from tqdm import tqdm
import numpy as np

from model import Bert

EPOCH = 1

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, criterion, train_loader):
    epoch_loss = 0

    start = time.time()
    model.to(device)
    model.train()

    for i, data in enumerate(tqdm(train_loader)):
        label, mask, sentence = data[0].to(device), data[1].to(device), data[2].to(device)
        optimizer.zero_grad()
        pred = model(sentence)
        print(pred)
        loss = criterion(pred, label)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    print('Loss: ', epoch_loss)
    end = time.time()
    print('Time: ', int(end-start))


def val(model, val_loader):
    model.to(device)
    model.eval()

    correct = 0
    wrong = 0
    for i, data in enumerate(val_loader):
        label, mask, sentence = data[0].to(device), data[1].to(device), data[2].to(device)
        with torch.no_grad():
            pred = model(sentence)
            print(pred)
            for j in range(len(pred)):
                pred_label = torch.argmax(pred[j])
                if pred_label == label[j]:
                    correct += 1
                else:
                    wrong += 1
        # label = label.to('cpu').numpy()
        # pred = pred.detach().cpu().numpy()
        # accuracy = flat_accuracy(pred, label)
    print('Correct: ', correct, '    Wrong: ', wrong)
    accuracy = correct / (correct+wrong)
    print('Accuracy: {:.2f}'.format(accuracy))


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


if __name__ == '__main__':
    train_path = './dbpedia.train'
    test_path = './dbpedia.test'
    data_set = DataSet(train_path)
    train_size = int(len(data_set)*0.9)
    val_size = len(data_set) - train_size

    train_set, val_set = random_split(data_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)

    print('Finish data loading')

    model = Bert()
    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss().to(device)

    for epoch in range(EPOCH):
        print('Epoch: ', epoch+1)
        train(model, optimizer, criterion, train_loader)
        val(model, val_loader)

