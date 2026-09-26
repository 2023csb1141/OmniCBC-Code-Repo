import os
import sys
import time
import numpy as np
import pandas as pd
import seaborn as sns
import IPython.display as ipd

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as patches
from matplotlib.colors import ListedColormap

from sklearn.preprocessing import StandardScaler

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device

data = pd.read_csv('')

data_unlabeled = data[data["Revised final diagnosis"].isna()]
data_labeled   = data[data["Revised final diagnosis"].notna()]

first_unlabeled = data_unlabeled.iloc[:, :58]
scaler = StandardScaler()
cols = first_unlabeled.columns
first_unlabeled = pd.DataFrame(scaler.fit_transform(first_unlabeled), columns = cols)

first_labeled = data_labeled.iloc[:, :58]
scaler = StandardScaler()
cols = first_labeled.columns
first_labeled = pd.DataFrame(scaler.fit_transform(first_labeled), columns = cols)

X_tensor = torch.from_numpy(first_unlabeled.to_numpy(dtype="float32"))

dataset = TensorDataset(X_tensor, X_tensor)
loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )


class AutoEncoder(nn.Module):

    def __init__(self, input_dim=58, latent_dim=16):
        super(AutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.GeLU(),

            nn.Linear(32, latent_dim))

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32),
            nn.GeLU(),

            nn.Linear(32, input_dim))


    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
model = AutoEncoder(input_dim=58, latent_dim=16).to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)
scheduler = StepLR(optimizer, step_size=10, gamma=0.5)

num_epochs = 250
Loss = []
Epochs = []

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for x, _ in loader:
        x = x.to(device)
        x_hat = model(x)
        loss = criterion(x_hat, x)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(loader)

    scheduler.step()

    current_lr = optimizer.param_groups[0]['lr']

    Loss.append(avg_loss)
    Epochs.append(epoch+1)

    if (epoch+1) % 10 == 0 or epoch == 0:
      print(f"Epoch [{epoch+1}/{num_epochs}]  Loss: {avg_loss:.6f}")

sns.lineplot(x=Epochs, y=Loss)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss vs Epochs')
plt.show()

model.eval()
with torch.no_grad():
    latent_unlabeled = model.encoder(X_tensor.to(device)).cpu().numpy()

print("Latent feature shape (Unlabeled):", latent_unlabeled.shape)

X_labeled = torch.from_numpy(first_labeled.to_numpy(dtype="float32"))

with torch.no_grad():
    latent_labeled = model.encoder(X_labeled.to(device)).cpu().numpy()

print("Latent feature shape (Labeled):", latent_labeled.shape)

df_unlabeled_ae = pd.DataFrame(scaler.fit_transform(latent_unlabeled))
df_labeled_ae = pd.DataFrame(scaler.fit_transform(latent_labeled))

final = pd.concat([df_unlabeled_ae, df_labeled_ae], axis=0, ignore_index=True)
final

final.describe()

final['Revised final diagnosis'] = data['Revised final diagnosis']

final.to_csv('autoencoded.csv', index=False)