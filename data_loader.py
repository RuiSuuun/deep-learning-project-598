import pandas as pd
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms

def load_data():
    data = pd.read_csv('data/sim.csv', sep=',')

    treated = data.loc[data['treat'] == 1 and data['momwhite'] == 1]
    control = data.loc[data['treat'] == 0]

    treated = treated.drop(columns=['momwhite', 'momblack', 'momhisp'])
    control = control.drop(columns=['momwhite', 'momblack', 'momhisp'])

    treated = torch.utils.data.DataLoader(treated, batch_size=16, shuffle=True)
    control = torch.utils.data.DataLoader(control, batch_size=16, shuffle=True)

    return treated, control