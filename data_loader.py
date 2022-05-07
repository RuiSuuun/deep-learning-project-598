import pandas as pd
import torch

def load_data():
    data = pd.read_csv('data/sim.csv', sep=',')

    treated = data.loc[(data['treat'] == 1) & (data['momwhite'] == 1)]
    control = data.loc[data['treat'] == 0]

    treated = treated.drop(columns=['momwhite', 'momblack', 'momhisp'])
    control = control.drop(columns=['momwhite', 'momblack', 'momhisp'])

    treated = torch.utils.data.DataLoader(treated, batch_size=128, shuffle=True)
    control = torch.utils.data.DataLoader(control, batch_size=128, shuffle=True)

    return treated, control