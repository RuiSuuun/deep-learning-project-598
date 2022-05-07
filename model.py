import torch
import torch.nn as nn


class Model(nn.Module):
    def __init__(self, in_channels=25):
        super(Model, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels,out_channels=16,kernel_size=(8),padding="same")
        self.actv1 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size = 1, stride = 1)

    def forward(self, x):
        print(x.shape)
        x = torch.reshape(x, (-1, x.shape[1], 1))
        print(x.shape)
        x = self.conv1(x)
        print(x.shape)
        x = self.actv1(x)
        print(x.shape)
        x = self.pool(x)
        print(x.shape)
        x = torch.reshape(x, (-1, x.shape[1] * x.shape[2]))
        return x
