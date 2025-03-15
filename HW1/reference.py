"""
You will need to validate your NN implementation using PyTorch. You can use any PyTorch functional or modules in this code.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class SingleLayerMLP(nn.Module):
    """constructing a single layer neural network with Pytorch"""
    def __init__(self, indim, outdim, hidden_layer=100):
        super(SingleLayerMLP, self).__init__()
        self.fc1 = nn.Linear(indim, hidden_layer)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_layer, outdim)
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x):
        """
        x shape (batch_size, indim)
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return self.softmax(x)


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        if isinstance(X, pd.DataFrame):
            self.X = X.to_numpy()
        else:
            self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length

model=None
criterion=None

def validate(loader):
    """takes in a dataloader, then returns the model loss and accuracy on this loader"""
    model.eval()
    total_loss = 0
    total_correct = 0
    total = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += y_batch.size(0)
            total_correct += (predicted == y_batch).sum().item()

    avg_loss = total_loss / len(loader)
    accuracy = total_correct / total
    return avg_loss, accuracy


if __name__ == "__main__":
    """The dataset loaders were provided for you.
    You need to implement your own training process.
    You need plot the loss and accuracies during the training process and test process. 
    """


    indim = 60
    outdim = 2
    hidden_dim = 100
    lr = 0.01
    batch_size = 64
    epochs = 500

    #dataset
    Xtrain = pd.read_csv("./data/X_train.csv")
    Ytrain = pd.read_csv("./data/y_train.csv")
    scaler = MinMaxScaler()
    Xtrain = pd.DataFrame(scaler.fit_transform(Xtrain), columns=Xtrain.columns).to_numpy()
    Ytrain = np.squeeze(Ytrain)
    m1, n1 = Xtrain.shape
    print(m1, n1)
    train_ds = DS(Xtrain, Ytrain)
    train_loader = DataLoader(train_ds, batch_size=batch_size)

    Xtest = pd.read_csv("./data/X_test.csv")
    Ytest = pd.read_csv("./data/y_test.csv").to_numpy()
    Xtest = pd.DataFrame(scaler.fit_transform(Xtest), columns=Xtest.columns.to_numpy())
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model
    model = SingleLayerMLP(indim, outdim, hidden_layer=hidden_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    #construct the training process
    train_loss, test_loss, train_acc, test_acc = [], [], [], []

    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        total_train_correct = 0
        total_train = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.float().to(device)
            y_batch = y_batch.long().to(device)
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += y_batch.size(0)
            total_train_correct += (predicted == y_batch).sum().item()

        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_correct / total_train
        avg_test_loss, avg_test_acc = validate(test_loader)

        train_loss.append(avg_train_loss)
        test_loss.append(avg_test_loss)
        train_acc.append(avg_train_acc)
        test_acc.append(avg_test_acc)

        print(f"Epoch [{epoch + 1}/{epochs}], "
              f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {avg_train_acc:.4f}, "
              f"Test Loss: {avg_test_loss:.4f}, Test Accuracy: {avg_test_acc:.4f}")

    size = 5
    Y1 = train_loss
    Y2 = test_loss
    X = [i + 1 for i in range(len(Y1))]

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.plot(X, Y1, label="train")
    ax.plot(X, Y2, label="test")
    ax.set_ylabel(r"Loss", fontsize=size)
    ax.set_xlabel(r"Epochs", fontsize=size)
    ax.set_title('PyTorch Loss', fontsize=size * 2)
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    ax.grid()
    ax.legend()
    plt.show()

    Y1 = train_acc
    Y2 = test_acc
    X = [i + 1 for i in range(len(Y1))]

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.plot(X, Y1, label="train")
    ax.plot(X, Y2, label="test")
    ax.set_ylabel(r"Accuracy", fontsize=size)
    ax.set_xlabel(r"Epochs", fontsize=size)
    ax.set_title('PyTorch Accuracy', fontsize=size * 2)
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    ax.grid()
    ax.legend()
    plt.show()