import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import numpy as np


def plot_loss_curves(epochs, loss_train, loss_valid):
    plt.plot(epochs, loss_train, label='Train Loss')
    plt.plot(epochs, loss_valid, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.show()

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
    y = (y > 5).astype(int)  # '1' for quality > 5, '0' otherwise

    train_x, ee_x, train_y, ee_y = train_test_split(X, y, test_size=0.3, random_state=42)
    valid_x, test_x, valid_y, test_y = train_test_split(ee_x, ee_y, test_size=0.5, random_state=42)

    # Convert to PyTorch tensors and explicitly cast to float
    train_x = torch.tensor(train_x.to_numpy(), dtype=torch.float)
    valid_x = torch.tensor(valid_x.to_numpy(), dtype=torch.float)
    test_x = torch.tensor(test_x.to_numpy(), dtype=torch.float)

    train_y = torch.tensor(train_y.to_numpy(), dtype=torch.long)  # Targets must be long for CrossEntropyLoss
    valid_y = torch.tensor(valid_y.to_numpy(), dtype=torch.long)
    test_y = torch.tensor(test_y.to_numpy(), dtype=torch.long)

    train_data = BasicDataset(train_x, train_y)
    valid_data = BasicDataset(valid_x, valid_y)
    test_data = BasicDataset(test_x, test_y)

    train_loader = DataLoader(dataset=train_data, batch_size=11)
    model = BasicMLP(n_inputs=train_x.shape[1], hidden_size=5, n_outputs=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.5)

    # Training loop

    epochs = 100

    loss_valid = []
    loss_train = []

    for epoch in range(epochs):
        model.train()
        # Batch the data
        epoch_loss_train = []
        for features, targets in train_loader:
            output = model.forward(features)  # Get model hypotheses
            # Calculate a loss
            loss = criterion(output, targets)
            epoch_loss_train.append(loss.item())
            optimizer.zero_grad()  # remove any previous losses
            loss.backward()  # Calculate the current losses backward through the MLP
            optimizer.step()  # Apply to the model itself
        # Calculate the training loss per epoch
        loss_train.append(sum(epoch_loss_train) / len(epoch_loss_train))

        # Per epoch, get validation performance
        model.eval()  # Puts the model in evaluation mode
        valid_hyp = model.forward(valid_x)
        loss_valid.append(criterion(valid_hyp, valid_y).item())
        c = torch.argmax(valid_hyp.data, dim=1)
        valid_accuracy = (c == valid_y).sum().item() / valid_x.shape[0]
        print('Epoch', epoch, 'train_loss', loss_train[-1], 'valid_loss', loss_valid[-1],
              'validation accuracy:', valid_accuracy)

    c = torch.argmax(valid_hyp, dim=1)
    valid_acc = (c == valid_y).sum().item() / valid_y.shape[0]
    print(valid_acc)