import random
import numpy as np
import torch
import os
import data_loader
from model import Model


# set seed
seed = 24
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

treated, control = data_loader.load_data()
model = Model()

# optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
# loss
criterion = torch.nn.CrossEntropyLoss()

def train(loader):
    model.train()
    for data in loader:
        optimizer.zero_grad()
        output = model(data.x)
        loss = criterion(output, data.y)
        loss.backward()
        optimizer.step()

def test(loader):
    model.eval()
    correct = 0
    for data in loader:
        output = model(data.x)
        correct += int((output == data.y).sum())  # Check against ground-truth labels.
    return correct / len(loader.dataset)

for epoch in range(200):
    train(treated)
    train_acc = test(treated)
    test_acc = test(control)
    print(f'Epoch: {epoch + 1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
