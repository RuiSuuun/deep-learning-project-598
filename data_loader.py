import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data.dataset import random_split
import torch

class CustomDataset(Dataset):
    def __init__(self, dataset):
        self.x = dataset.drop(columns=['treat'])
        self.y = dataset['treat']

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

def load_data():
    def collate_fn(data):
        features, treatment = zip(*data)

        y = torch.tensor(treatment, dtype=torch.bool)
        x = torch.tensor(features, dtype=torch.float)
        return x, y

    data = pd.read_csv('data/sim.csv', sep=',')

    treated = data.loc[(data['treat'] == 1) & (data['momwhite'] == 1)]
    control = data.loc[data['treat'] == 0]
    dataset = pd.concat([treated, control])

    dataset = dataset.drop(columns=['momwhite', 'momblack', 'momhisp', 'Unnamed: 0'])
    dataset = dataset.reset_index(drop=True)

    dataset = CustomDataset(dataset)

    split = int(len(dataset)*0.8)
    lengths = [split, len(dataset) - split]
    train_dataset, val_dataset = random_split(dataset, lengths)

    print("Length of train dataset:", len(train_dataset))
    print("Length of val dataset:", len(val_dataset))

    train_dataset = torch.utils.data.DataLoader(train_dataset, batch_size=128, collate_fn=collate_fn)
    val_dataset = torch.utils.data.DataLoader(val_dataset, batch_size=128, collate_fn=collate_fn)

    return train_dataset, val_dataset