import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class BasicMLP(torch.nn.Module):
    def __init__(self, n_inputs, hidden_size, n_outputs):
        super(BasicMLP, self).__init__()

        self.inputs = n_inputs
        self.hidden = hidden_size
        self.outputs = n_outputs

        self.fc1 = torch.nn.Linear(self.inputs, self.hidden)
        self.fc2 = torch.nn.Linear(self.hidden, self.outputs)
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu(out)
        out = self.fc2(out)
        return out


class BasicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    df = pd.read_csv('winequality-red.csv')
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    train_x, ee_x, train_y, ee_y = train_test_split(X, y, test_size=0.3, random_state=42)
    valid_x, test_x, valid_y, test_y = train_test_split(ee_x, ee_y, test_size=0.5, random_state=42)

    train_x = torch.tensor(train_x.to_numpy())
    train_x = train_x.float()
    test_x, valid_x = map(torch.tensor, (test_x.to_numpy(), valid_x.to_numpy()))
    test_x = test_x.float()
    valid_x = valid_x.float()
    train_y, valid_y, test_y = map(torch.tensor, (train_y.to_numpy(), valid_y.to_numpy(), test_y.to_numpy()))

    train_data = BasicDataset(train_x, train_y)
    valid_data = BasicDataset(valid_x, valid_y)
    test_data = BasicDataset(test_x, test_y)
    wine_basic_train = BasicDataset(train_x, train_y)

    train_loader = DataLoader(dataset=wine_basic_train, batch_size=11)
    model = BasicMLP(n_inputs=train_x.shape[1], hidden_size=5, n_outputs=1)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.7)

    # Training loop

    epochs = 100
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)

        model.eval()
        valid_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for features, targets in valid_loader:
                outputs = model(features)
                loss = criterion(outputs, targets)
                valid_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        valid_loss /= len(valid_loader)
        accuracy = 100 * correct / total

        print(
            f'Epoch {epoch}: Train Loss = {train_loss}, Validation Loss = {valid_loss}, Validation Accuracy = {accuracy}%')

