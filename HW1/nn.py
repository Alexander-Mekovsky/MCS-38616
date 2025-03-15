"""
You will need to implement a single layer neural network from scratch.

IMPORTANT: DO NOT change any function signatures
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import torch
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple, Dict


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class Transform(object):
    """
    This is the base class. You do not need to change anything.
    Read the comments in this class carefully.
    """
    def __init__(self):
        """
        Initialize any parameters
        """
        pass

    def forward(self, x):
        """
        x should be passed as column vectors
        """
        pass

    def backward(self, grad_wrt_out):
        """
        Compute and save the gradients wrt the parameters for step()
        Return grad_wrt_x which will be the grad_wrt_out for previous Transform
        """
        pass

    def step(self):
        """
        Apply gradients to update the parameters
        """
        pass

    def zerograd(self):
        """
        This is used to Reset the gradients.
        Usually called before backward()
        """
        pass


class ReLU(Transform):
    def __init__(self):
        super(ReLU, self).__init__()

    def forward(self, x):
        self.x = x
        return torch.maximum(torch.tensor(0.0, device=x.device), x)

    def backward(self, grad_wrt_out):
        grad_wrt_input = grad_wrt_out * (self.x > 0).T
        return grad_wrt_input


class LinearMap(Transform):
    def __init__(self, indim, outdim, lr=0.01):
        """
        indim: input dimension
        outdim: output dimension
        lr: learning rate
        """
        super(LinearMap, self).__init__()
        self.weights = 0.01 *torch.rand((outdim, indim), dtype=torch.float64, requires_grad=True, device=device)
        self.bias = 0.01 * torch.rand((outdim, 1), dtype=torch.float64, requires_grad=True, device=device)
        self.lr = lr


    def forward(self, x):
        self.x = x
        out = torch.mm(self.weights, x)
        out += self.bias
        return out


    def backward(self, grad_wrt_out):
        grad_wrt_out = grad_wrt_out.T
        grad_wrt_weights = torch.mm(grad_wrt_out, self.x.T)
        grad_wrt_bias = torch.sum(grad_wrt_out, axis=1, keepdims=True)
        grad_wrt_input = torch.mm(self.weights.T, grad_wrt_out)
        self.weights.grad = grad_wrt_weights
        self.bias.grad = grad_wrt_bias
        return grad_wrt_weights, grad_wrt_bias, grad_wrt_input
        #compute grad_wrt_weights
        
        #compute grad_wrt_bias
        
        #compute & return grad_wrt_input
        


    def step(self):
        self.weights -= self.lr * self.weights.grad
        self.bias -= self.lr * self.bias.grad
        self.weights.grad.zero_()
        self.bias.grad.zero_()


class SoftmaxCrossEntropyLoss(object):
    def forward(self, logits, labels):
        """
        logits are pre-softmax scores, labels are one-hot labels of given inputs
        logits and labels are in the shape of (num_classes, batch_size)
        returns loss as a scalar (i.e. mean value of the batch_size loss)
        """
        self.logits = logits
        self.labels = labels
        exp_logits = torch.exp(logits - torch.max(logits, dim=0, keepdim=True)[0])
        self.softmax_probs = exp_logits / torch.sum(exp_logits, dim=0, keepdim=True)
        self.softmax_probs = self.softmax_probs.T
        batch_size = logits.shape[1]
        log_likelihood = -torch.log(self.softmax_probs[torch.arange(batch_size), torch.argmax(labels, dim=1)])
        loss = torch.mean(log_likelihood)
        return loss


    def backward(self):
        """
        return grad_wrt_logits shape (num_classes, batch_size)
        (don't forget to divide by batch_size because your loss is a mean)
        """
        batch_size = self.logits.shape[1]
        grad_wrt_logits = (self.softmax_probs - self.labels) / batch_size
        return grad_wrt_logits

    def getAccu(self):
        """
        return accuracy here
        """
        predictions = torch.argmax(self.softmax_probs, dim=1)
        correct = torch.sum(predictions == torch.argmax(self.labels, dim=1)).item()
        accuracy = correct / self.labels.shape[0]
        return accuracy


class SingleLayerMLP(Transform):
    """constructing a single layer neural network with the previous functions"""
    def __init__(self, indim, outdim, hidden_layer=100, lr=0.01):
        super(SingleLayerMLP, self).__init__()
        self.linear = LinearMap(indim, outdim, lr)
        self.relu = ReLU()
        self.loss_fn = SoftmaxCrossEntropyLoss()


    def forward(self, x):
        """
        x shape (indim, batch_size)
        return the presoftmax logits shape(outdim, batch_size)
        """
        logits = self.linear.forward(x)
        logits = self.relu.forward(logits)
        return logits


    def backward(self, grad_wrt_out):
        """
        grad_wrt_out shape (outdim, batch_size)
        calculate the gradients wrt the parameters
        """
        grad_wrt_relu = self.relu.backward(grad_wrt_out)
        grad_wrt_weights, grad_wrt_bias, grad_wrt_input = self.linear.backward(grad_wrt_relu)
        return grad_wrt_weights, grad_wrt_bias, grad_wrt_input

    
    def step(self):
        """update model parameters"""
        self.linear.step()


class DS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.length = len(X)
        self.X = X
        self.Y = Y

    def __getitem__(self, idx):
        x = self.X[idx, :]
        y = self.Y[idx]
        return (x, y)

    def __len__(self):
        return self.length

def labels2onehot(labels: np.ndarray):
    return np.array([[i==lab for i in range(2)] for lab in labels]).astype(int)

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
    Ytest = pd.read_csv("./data/y_test.csv")
    Xtest = pd.DataFrame(scaler.fit_transform(Xtest), columns=Xtest.columns).to_numpy()
    Ytest = np.squeeze(Ytest)
    m2, n2 = Xtest.shape
    print(m1, n2)
    test_ds = DS(Xtest, Ytest)
    test_loader = DataLoader(test_ds, batch_size=batch_size)

    #construct the model
    model = SingleLayerMLP(indim, outdim, lr=lr)
    #construct the training process
    train_loss, test_loss, train_acc, test_acc = [], [], [], []
    for epoch in range(epochs):
        total_train_loss = 0
        total_train_acc = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.T.to(device)
            y_batch = y_batch.to(device)
            y_batch_onehot = labels2onehot(y_batch.numpy())
            y_batch_onehot = torch.tensor(y_batch_onehot, dtype=torch.float64, device=device)
            logits = model.forward(X_batch)
            loss = model.loss_fn.forward(logits, y_batch_onehot)
            total_train_loss += loss.item()
            model.backward(model.loss_fn.backward())
            model.step()
            train_accuracy = model.loss_fn.getAccu()
            total_train_acc += train_accuracy
        avg_train_loss = total_train_loss / len(train_loader)
        avg_train_acc = total_train_acc / len(train_loader)
        total_test_loss = 0
        total_test_acc = 0
        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                X_batch = X_batch.T.to(device)
                y_batch = y_batch.to(device)
                y_batch_onehot = labels2onehot(y_batch.numpy())
                y_batch_onehot = torch.tensor(y_batch_onehot, dtype=torch.float64, device=device)
                logits = model.forward(X_batch)
                loss = model.loss_fn.forward(logits, y_batch_onehot)
                total_test_loss += loss.item()
                test_accuracy = model.loss_fn.getAccu()
                total_test_acc += test_accuracy

        avg_test_loss = total_test_loss / len(test_loader)
        avg_test_acc = total_test_acc / len(test_loader)

        train_loss.append(avg_train_loss)
        test_loss.append(avg_test_loss)
        train_acc.append(avg_train_acc)
        test_acc.append(avg_test_acc)

        print(f'Epoch [{epoch + 1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Train Acc: {avg_train_acc:.4f}, '
              f'Test Loss: {avg_test_loss:.4f}, Test Acc: {avg_test_acc:.4f}')

    size = 5
    Y1 = train_loss
    Y2 = test_loss
    X = [i + 1 for i in range(len(Y1))]

    fig, ax = plt.subplots(figsize=(4, 3), dpi=300)
    ax.plot(X, Y1, label="train")
    ax.plot(X, Y2, label="test")
    ax.set_ylabel(r"Loss", fontsize=size)
    ax.set_xlabel(r"Epochs", fontsize=size)
    ax.set_title('Homemade Loss', fontsize=size*2)
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
    ax.set_title('Homemade Accuracy', fontsize=size * 2)
    plt.xticks(fontsize=size)
    plt.yticks(fontsize=size)
    ax.grid()
    ax.legend()
    plt.show()

