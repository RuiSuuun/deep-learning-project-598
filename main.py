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

train_dataset, val_dataset, dataset_overall = data_loader.load_data()
model = Model(dataset_overall)
print(model)

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
        input_data = generate_random_coefficient_vector() * data.x
        output_data = model(input_data)
        correct += int((output_data == data.y).sum())  # Check against ground-truth labels.

        pehe = torch.pow(PEHE(output_data, data.y), 0.5)
        ate = ATE(output_data, data.y)
    return correct / len(loader.dataset), pehe, ate

def generate_random_coefficient_vector():
    return np.random.choice(np.arange(0, 5), p=[0.5, 0.2, 0.15, 0.1, 0.05])

def PEHE(prediction, actual):
    return torch.mean(torch.pow(prediction - actual, 2))

def ATE(prediction, actual):
    return torch.mean(torch.abs(prediction - actual))

for epoch in range(1000):
    train(train_dataset)
    train_acc, _, _ = test(train_dataset)
    test_acc, pehe, ate = test(val_dataset)
    print(f'Epoch: {epoch + 1:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Epoch: {epoch + 1:03d}, PEHE: {pehe:.2f}, ATE: {ate:.2f}')
