import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels=25):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels=16,kernel_size=(8),padding="same")
        self.actv1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size = 1, stride = 1)
        self.linear = nn.Linear(in_features=16, out_features=8)
        self.actv2 = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):
        x = torch.reshape(x, (-1, x.shape[1], 1))
        x = self.conv1(x)
        x = self.actv1(x)
        x = self.pool(x)
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2]))
        x = self.linear(x)
        x = self.actv2(x)
        x = self.dropout(x)
        return x
