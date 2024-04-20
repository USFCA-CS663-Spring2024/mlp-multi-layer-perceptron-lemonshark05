import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# Define the MLP model
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

def plot_loss_curves(epochs, loss_train, loss_valid):
    plt.plot(epochs, loss_train, label='Train Loss')
    plt.plot(epochs, loss_valid, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Cross Entropy Loss')
    plt.legend()
    plt.show()


class NewBasicMLP(torch.nn.Module):
    def __init__(self, n_inputs, hidden_size, n_outputs):
        super(NewBasicMLP, self).__init__()

        self.inputs = n_inputs
        self.hidden = hidden_size
        self.outputs = n_outputs

        self.fc1 = torch.nn.Linear(self.inputs, self.hidden)  # First Hidden Layer
        self.fc2 = torch.nn.Linear(self.hidden, self.hidden)  # Second Hidden Layer
        self.fc3 = torch.nn.Linear(self.hidden, self.outputs)  # Output Layer
        self.relu = torch.nn.ReLU()

    def forward(self, X):
        out = self.fc1(X)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.relu(out)
        out = self.fc3(out)
        return out

class BasicDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y.view(-1, 1)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)

if __name__ == '__main__':
    df = pd.read_csv('winequality-red.csv')
    X, y = df.iloc[:, :-1], df.iloc[:, -1]

    train_x, ee_x, train_y, ee_y = train_test_split(X, y, test_size=0.3, random_state=42)
    valid_x, test_x, valid_y, test_y = train_test_split(ee_x, ee_y, test_size=0.5, random_state=42)

    train_x = torch.tensor(train_x.to_numpy(), dtype=torch.float)
    valid_x = torch.tensor(valid_x.to_numpy(), dtype=torch.float)
    test_x = torch.tensor(test_x.to_numpy(), dtype=torch.float)
    train_y = torch.tensor(train_y.to_numpy(), dtype=torch.float)
    valid_y = torch.tensor(valid_y.to_numpy(), dtype=torch.float)
    test_y = torch.tensor(test_y.to_numpy(), dtype=torch.float)

    train_data = BasicDataset(train_x, train_y)
    valid_data = BasicDataset(valid_x, valid_y)
    test_data = BasicDataset(test_x, test_y)

    train_loader = DataLoader(train_data, batch_size=11, shuffle=True)
    model = BasicMLP(n_inputs=train_x.shape[1], hidden_size=5)
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.0001, momentum=0.7)

    # Training loop

    epochs = 100
    train_losses = []
    valid_losses = []

    tolerance = 1
    total_valid_accuracy = 0

    for epoch in range(epochs):
        model.train()
        batch_losses = []
        for features, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets)
            loss.backward()  # Calculate the current losses backward through the MLP
            optimizer.step()  # Apply to the model itself
            batch_losses.append(loss.item())

        # Calculate the training loss per epoch
        train_loss = np.mean(batch_losses)
        train_losses.append(train_loss)

        model.eval()
        valid_hyp = model(valid_x)
        valid_loss = criterion(valid_hyp, valid_y)
        valid_losses.append(valid_loss.item())
        # Calculate validation accuracy
        valid_accuracy = ((valid_hyp - valid_y).abs() <= tolerance).float().mean().item()
        total_valid_accuracy += valid_accuracy

        print(f'Epoch {epoch}: Train Loss = {train_loss}, Validation Loss = {valid_loss}, Validation Accuracy = {valid_accuracy}')
    plot_loss_curves(np.linspace(1, epochs, epochs).astype(int), train_losses, valid_losses)
    valid_acc = total_valid_accuracy / epochs
    print(valid_acc)