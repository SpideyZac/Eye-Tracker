import os
from h11 import Data
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class EyeTrackerDataset(Dataset):
    def __init__(self, csv_file):
        self.er_frame = pd.read_csv(csv_file)
        self.m_x_a = np.asarray(self.er_frame.iloc[:, 0])
        self.m_y_a = np.asarray(self.er_frame.iloc[:, 1])
        self.h_r_a = np.asarray(self.er_frame.iloc[:, 2])
        self.v_r_a = np.asarray(self.er_frame.iloc[:, 3])

    def __len__(self):
        return len(self.er_frame)

    def __getitem__(self, index):
        m_x = self.m_x_a[index]
        m_y = self.m_y_a[index]
        h_r = self.h_r_a[index]
        v_r = self.v_r_a[index]

        return torch.Tensor([h_r, v_r]), torch.Tensor([m_x, m_y])

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        x = self.fc4(x)

        return x

trainset = DataLoader(EyeTrackerDataset("data/data.csv"), batch_size=10, shuffle=True)
net = Net()
try:
    net = torch.load("model")
except:
    net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)

EPOCHS = 500

for epoch in range(EPOCHS):
    for data in trainset:
        X, y = data
        net.zero_grad()
        output = net(X)
        loss = F.l1_loss(output, y)
        loss.backward()
        optimizer.step()
    print(loss)
        
torch.save(net, "model")