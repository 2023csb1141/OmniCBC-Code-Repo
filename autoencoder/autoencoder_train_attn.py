import math
import os
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import time
import matplotlib.pyplot as plt
import shutil
import random
import numpy as np

### Dataset
class TabularDataset(Dataset):
    def __init__(self, X):
        self.X = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx]


### Model

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        attn_out, _ = self.attn(x, x, x)
        return self.norm(x + attn_out)
    
class EncoderBlock(nn.Module):
    def __init__(self,):
        super().__init__()
        self.fc = nn.Linear(58, 64)
        self.relu1 = nn.ReLU()

        self.l1 = nn.Linear(64, 64)
        self.batchnorm1 = nn.BatchNorm1d(64)

        self.attn1 = SelfAttentionBlock(64)
        self.gelu1 = nn.GELU()

        self.l2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()

        self.l3 = nn.Linear(32, 32)
        self.batchnorm2 = nn.BatchNorm1d(32)

        self.attn2 = SelfAttentionBlock(32)
        self.gelu2 = nn.GELU()

        self.l4 = nn.Linear(32, 16)
        self.relu3 = nn.ReLU()
        self.l5 = nn.Linear(16, 16)
        self.batchnorm3 = nn.BatchNorm1d(16)
        self.gelu3 = nn.GELU()

    def forward(self, x):
        x = self.fc(x)
        x = self.relu1(x)

        x = self.l1(x)
        x = self.batchnorm1(x)
        x = self.attn1(x.unsqueeze(1)).squeeze(1)
        x = self.gelu1(x)

        x = self.l2(x)
        x = self.relu2(x)

        x = self.l3(x)
        x = self.batchnorm2(x)
        x = self.attn2(x.unsqueeze(1)).squeeze(1)
        x = self.gelu2(x)

        x = self.l4(x)
        x = self.relu3(x)

        x = self.l5(x)
        x = self.batchnorm3(x)
        x = self.gelu3(x)

        return x

class DecoderBlock(nn.Module):
    def __init__(self,):
        super().__init__()
        self.l1 = nn.Linear(16, 16)
        self.batchnorm1 = nn.BatchNorm1d(16)
        self.gelu1 = nn.GELU()

        self.l2 = nn.Linear(16, 32)
        self.relu1 = nn.ReLU()

        self.l3 = nn.Linear(32, 32)
        self.batchnorm2 = nn.BatchNorm1d(32)
        self.gelu2 = nn.GELU()

        self.l4 = nn.Linear(32, 64)
        self.relu2 = nn.ReLU()
        
        self.l5 = nn.Linear(64, 64)
        self.gelu3 = nn.GELU()

        self.fc = nn.Linear(64, 58)
        self.relu3 = nn.ReLU()

    def forward(self, x):
        x = self.l1(x)
        x = self.batchnorm1(x)
        x = self.gelu1(x)

        x = self.l2(x)
        x = self.relu1(x)

        x = self.l3(x)
        x = self.batchnorm2(x)
        x = self.gelu2(x)

        x = self.l4(x)
        x = self.relu2(x)

        x = self.l5(x)
        x = self.gelu3(x)

        x = self.fc(x)
        x = self.relu3(x)

        return x

class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = EncoderBlock()
        self.decoder = DecoderBlock()

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat, z


class CustomLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def forward(self, x, y):
        mse = self.mse_loss(x, y)
        l1 = self.l1_loss(x, y)

        return (mse, l1, mse + l1)


def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


### training function
def main():
    data_csv_path = ""
    results_dir = ""
    num_epochs = 1000
    seed = 1337

    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)
    CKPT_LATEST = os.path.join(results_dir, "latest.pth")
    CKPT_BEST = os.path.join(results_dir, "best.pth")
    LOG_CSV = os.path.join(results_dir, "training_log.csv")
    LOSS_PLOT = os.path.join(results_dir, "epoch_vs_loss.png")
    BATCH_SIZE = 256
    LEARNING_RATE = 1e-4
    
    with open(LOG_CSV, 'w') as f:
        f.write("Epoch,MSELoss,L1Loss,Total Loss,Epoch Time\n")

    print(f"Loading data from {data_csv_path}...")

    data = pd.read_csv(data_csv_path)

    print(f"Preparing dataset...")

    features = data.iloc[:, :58]
    labels = data["Revised final diagnosis"]

    # train on only UNLABELED patients
    features = features[labels.isna()].reset_index(drop=True)

    scaler = StandardScaler()
    X = scaler.fit_transform(features)

    dataset = TabularDataset(X)
    g = torch.Generator()
    g.manual_seed(42)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=4, 
        persistent_workers=True, 
        worker_init_fn=seed_worker, 
        generator=g
    )

    ### Model 
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    model = AutoEncoder().to(device)
    criterion = CustomLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.0, total_iters=num_epochs)

    ### Training
    print(f"Starting training...")
    best_loss = float("inf")

    log_records = []
    training_start_time = time.time()

    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        model.train()
        running_loss = 0.0
        epoch_mse = 0.0
        epoch_l1 = 0.0

        for x in loader:
            x = x.to(device)

            optimizer.zero_grad()
            x_hat, _ = model(x)
            loss_tup = criterion(x_hat, x)
            loss = loss_tup[2]
            l1 = loss_tup[1]
            mse = loss_tup[0]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_mse += mse.item()
            epoch_l1 += l1.item()

        scheduler.step()

        epoch_loss = running_loss / math.ceil(len(dataset)/BATCH_SIZE)
        print(math.ceil(len(dataset)/BATCH_SIZE))
        epoch_mse = epoch_mse / math.ceil(len(dataset)/BATCH_SIZE)
        epoch_l1 = epoch_l1 / math.ceil(len(dataset)/BATCH_SIZE)
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.6f} | Epoch time: {epoch_time:.2f}s ")

        log_records.append({
            "epoch": epoch + 1,
            "total_loss": epoch_loss,
            "mse_loss": epoch_mse,
            "l1_loss": epoch_l1,
            "epoch_time": epoch_time
        })

        with open(LOG_CSV, 'a') as f:
            f.write(f"{epoch + 1},{epoch_mse:.6f},{epoch_l1:.6f},{epoch_loss:.6f},{epoch_time:.2f}\n")

        torch.save({
            "epoch": epoch + 1,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss": epoch_loss,
            "scaler": scaler
        }, CKPT_LATEST)

        # if (epoch+1)%100==0:
        #     shutil.copy(CKPT_BEST, os.path.join(results_dir, f"best_upto_{epoch+1}.pth"))
            
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                "epoch": epoch + 1,
                "model_state": model.state_dict(),
                "loss": epoch_loss,
                "scaler": scaler
            }, CKPT_BEST)


    # -------------------------
    # Plot Epoch vs Loss
    # -------------------------
    log_df = pd.DataFrame(log_records)

    plt.figure()
    plt.plot(log_df["epoch"], log_df["total_loss"], label="Total Loss")
    plt.plot(log_df["epoch"], log_df["mse_loss"], label="MSE Loss")
    plt.plot(log_df["epoch"], log_df["l1_loss"], label="L1 Loss")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Reconstruction Loss (MSE)")
    plt.title("Autoencoder Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=300)
    plt.close()


if __name__ == "__main__":
    main()
