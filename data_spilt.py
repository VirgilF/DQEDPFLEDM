import numpy as np
import torch
import torch.utils.data
from torch.distributions.dirichlet import Dirichlet
from PIL import Image
from torch.utils.data import DataLoader
import torch.utils.data
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.nn.functional as F
import pandas as pd
import os
import random
import math
from sklearn.preprocessing import StandardScaler
import warnings
from model import MLP
from set import args
warnings.filterwarnings("ignore")

class MYDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self._x = x
        self._y = y
        self._len = len(x)

    def __getitem__(self, item):
        img, label = self._x[item], self._y[item]
        return torch.tensor(img), torch.tensor(label)

    def __len__(self):
        return self._len




def dirichlet_split_noniid(train_labels, alpha, n_clients):
    n_classes = train_labels.max() + 1
    label_distribution = Dirichlet(torch.full((n_clients,), alpha).float()).sample()

    class_idcs = []
    for value in range(7):
        indices = torch.nonzero(torch.eq(train_labels, value)).squeeze()
        class_idcs.append(indices)
    client_idcs = [[] for _ in range((n_clients))]
    for c in class_idcs:
        total_size = len(c)
        splits = (label_distribution * total_size).int()
        splits[-1] = total_size - splits[:-1].sum()
        idcs = torch.split(c, splits.tolist())
        for i, idx in enumerate(idcs):
            client_idcs[i] += [idcs[i]]

    client_idcs = [torch.cat(idcs) for idcs in client_idcs]
    return client_idcs


data=pd.read_csv(r'C:\Users\fcy\Desktop\FL+sv+dp\FL+sv+dp\ETDataset-main\ETDataset-main\ETT-small\ETTm1.csv')
data['date'] = pd.to_datetime(data['date'],format='%Y-%m-%d %H:%M:%S')

data['year'] = data['date'].dt.year
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour + data['date'].dt.minute / 60 + data['date'].dt.second / 3600
data.drop('date', axis=1, inplace=True)

data['datetime'] = pd.to_datetime(data[['year', 'month', 'day', 'hour']])
data['season'] = (data['month'] % 12 + 3) // 3
data['weekday'] = (data['datetime'].dt.dayofweek < 5).astype(float)

data['hour_of_day'] = data['datetime'].dt.hour

scaler = StandardScaler()
scaled_features = scaler.fit_transform(data[['HUFL', 'HULL', 'MUFL', 'MULL', 'LUFL', 'LULL', 'season', 'weekday', 'hour_of_day']])

label=np.array(data['OT'].values)
data.drop('OT',axis=1, inplace=True)

Data=np.array(scaled_features)
train_size = int(0.8 * len(Data))
train_data = Data[:train_size]
test_data = Data[train_size:]
train_labels = label[:train_size]
test_labels = label[train_size:]
print(train_data.shape)
print(train_labels.shape)
print(test_data.shape)
print(test_labels.shape)

splitted_data = np.array_split(train_data, args.user_num)
splitted_label = np.array_split(train_labels, args.user_num)
test_dataset = MYDataset(test_data, test_labels)
client_dataloder=[]

def loader():
    for i in range(args.user_num):
        train_dataset=MYDataset(splitted_data[i], splitted_label[i])
        client_dataloder.append(torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True))
    return client_dataloder

def test():
    test_loader=torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return test_loader




