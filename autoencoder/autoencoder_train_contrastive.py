import math
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import time
import random
import matplotlib.pyplot as plt
import shutil
from collections import defaultdict

### Dataset
class TabularDataset(Dataset):
    def __init__(self, X, ids):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.ids = ids

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.ids[idx]


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


class CustomLossEuclidean(nn.Module):
    def __init__(self, contrastive_weight=1.0, max_sampling_neighbours=15):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = nn.MSELoss()  # Using MSE as a proxy for Euclidean distance
        self.l1_loss = nn.L1Loss()
        self.contrastive_weight = contrastive_weight
        self.max_neighbors = max_sampling_neighbours

    def forward(self, x_reconstructed, x_latent, x_org, Z_all, patient_ids, graph):

        mse = self.mse_loss(x_reconstructed, x_org)
        l1 = self.l1_loss(x_reconstructed, x_org)

        # print(f"All shapes are... \n0: x_reconstructed: {x_reconstructed.shape}, \n1: x_latent: {x_latent.shape}, \n2: x_org: {x_org.shape}, \n3: Z_all: {Z_all.shape}, \n4: patient_ids: {patient_ids.shape}")

        zi_list = []
        zj_list = []

        for idx, node in enumerate(patient_ids):
            neighbors = graph.get(node, [])
            if len(neighbors) == 0:
                continue

            if len(neighbors) > self.max_neighbors:
                neighbors = random.sample(neighbors, self.max_neighbors)

            zi = x_latent[idx]

            for nbr, _ in neighbors:
                zi_list.append(zi)
                zj_list.append(Z_all[nbr - 1])

        if len(zi_list) == 0:
            contrastive = torch.tensor(0.0, device=x_latent.device)
        else:
            zi_all = torch.stack(zi_list)      # [N_pairs, latent_dim]
            zj_all = torch.stack(zj_list)

            contrastive = self.contrastive_loss(zi_all, zj_all)

        total = mse + l1 + self.contrastive_weight * contrastive
        return mse, l1, contrastive, total
    

class CustomLossCosine(nn.Module):
    def __init__(self, contrastive_weight=1.0, max_sampling_neighbours=15):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.contrastive_loss = nn.CosineEmbeddingLoss()
        self.l1_loss = nn.L1Loss()
        self.contrastive_weight = contrastive_weight
        self.max_neighbors = max_sampling_neighbours

    def forward(self, x_reconstructed, x_latent, x_org, Z_all, patient_ids, graph, patient_id_to_int):

        mse = self.mse_loss(x_reconstructed, x_org)
        l1 = self.l1_loss(x_reconstructed, x_org)


        zi_list = []
        zj_list = []

        for idx, node in enumerate(patient_ids):
            neighbors = graph.get(node, [])
            if len(neighbors) == 0:
                continue

            if len(neighbors) > self.max_neighbors:
                neighbors = random.sample(neighbors, self.max_neighbors)

            zi = x_latent[idx]

            # print(f"Node: {node}, Neighbors: {neighbors}, Zi shape: {zi.shape}")

            for nbr, wt in neighbors:
                zi_list.append(zi)
                zj_list.append(Z_all[patient_id_to_int[nbr]])

        if len(zi_list) == 0:
            contrastive = torch.tensor(0.0, device=x_latent.device)
        else:
            zi_all = torch.stack(zi_list)      # [N_pairs, latent_dim]
            zj_all = torch.stack(zj_list)
            targets = torch.ones(
                zi_all.size(0), device=x_latent.device
            )

            contrastive = self.contrastive_loss(zi_all, zj_all, targets)

        total = mse + l1 + self.contrastive_weight * contrastive
        return mse, l1, contrastive, total
    

def compute_all_embeddings(model, X_tensor):
    with torch.no_grad():
        _, latent = model(X_tensor)
    return latent
        

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
def main(global_connection_graph_dict):

    data_csv_path = ""
    results_dir = ""
    loss_wgt = 0.2
    num_epochs = 1000
    seed = 1337

    set_seed(seed)
    os.makedirs(results_dir, exist_ok=True)
    CKPT_LATEST = os.path.join(results_dir, "latest.pth")
    CKPT_BEST = os.path.join(results_dir, "best.pth")
    LOG_CSV = os.path.join(results_dir, "training_log.csv")
    LOSS_PLOT = os.path.join(results_dir, "epoch_vs_loss.png")
    BATCH_SIZE = 512
    LEARNING_RATE = 1e-4
    
    with open(LOG_CSV, 'w') as f:
        f.write("Epoch,MSELoss,L1Loss, Contrastive Loss,Total Loss,Epoch Time\n")

    print(f"Loading data from {data_csv_path}...")

    df = pd.read_csv(data_csv_path)
    print(f"Dataframe shape: {df.shape}")

    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    print(f"Dataframe shape after shuffling: {df.shape}")

    main_feature_columns = ['WBC(10^3/uL)', 'RBC(10^6/uL)', 'HGB(g/dL)', 'HCT(%)', 'MCV(fL)', 'MCH(pg)', 'MCHC(g/dL)', 'PLT(10^3/uL)', 'NRBC#(10^3/uL)', 'NRBC%(%)', 'NEUT#(10^3/uL)', 'LYMPH#(10^3/uL)', 'MONO#(10^3/uL)', 'EO#(10^3/uL)', 'BASO#(10^3/uL)', 'NEUT%(%)', 'LYMPH%(%)', 'MONO%(%)', 'EO%(%)', 'BASO%(%)', 'IG#(10^3/uL)', 'IG%(%)', '[PLT-I(10^3/uL)]', 'MicroR(%)', 'MacroR(%)', '[TNC(10^3/uL)]', '[WBC-N(10^3/uL)]', '[TNC-N(10^3/uL)]', '[BA-N#(10^3/uL)]', '[BA-N%(%)]', '[WBC-D(10^3/uL)]', '[TNC-D(10^3/uL)]', '[NEUT#&(10^3/uL)]', '[NEUT%&(%)]', '[LYMP#&(10^3/uL)]', '[LYMP%&(%)]', '[HFLC#(10^3/uL)]', '[HFLC%(%)]', '[BA-D#(10^3/uL)]', '[BA-D%(%)]', '[NE-SSC(ch)]', '[NE-SFL(ch)]', '[NE-FSC(ch)]', '[LY-X(ch)]', '[LY-Y(ch)]', '[LY-Z(ch)]', '[MO-X(ch)]', '[MO-Y(ch)]', '[MO-Z(ch)]', '[NE-WX]', '[NE-WY]', '[NE-WZ]', '[LY-WX]', '[LY-WY]', '[LY-WZ]', '[MO-WX]', '[MO-WY]', '[MO-WZ]']
    patient_id_column = ['pseudo_patient_id']

    df = df[patient_id_column + main_feature_columns]

    scaler = StandardScaler()
    df[main_feature_columns] = scaler.fit_transform(df[main_feature_columns])

    patient_id_to_int = {pid: idx for idx, pid in enumerate(df['pseudo_patient_id'])}
    df['created_id'] = df['pseudo_patient_id'].map(patient_id_to_int)

    X = df[main_feature_columns].values
    all_patient_ids = df[patient_id_column].values
    all_patient_ids = all_patient_ids.squeeze()

    print(f"Dataset shape: {X.shape}, Patient IDs shape: {all_patient_ids.shape}")
    print(f"Dataset type: {type(X)}, Patient IDs type: {type(all_patient_ids)}")

    dataset = TabularDataset(X, all_patient_ids)
    g = torch.Generator()
    g.manual_seed(seed)
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

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

    model = AutoEncoder().to(device)
    # criterion = CustomLossEuclidean(contrastive_weight=loss_wgt)
    criterion = CustomLossCosine(contrastive_weight=loss_wgt)
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
        epoch_contrastive = 0.0

        # print(f"{X_tensor.shape}")
        Z_all = compute_all_embeddings(model, X_tensor)

        for x, patient_id in loader:
            x = x.to(device)
            # patient_id = patient_id.to(device)

            optimizer.zero_grad()
            x_hat, latent_x = model(x)
            loss_tup = criterion(x_hat, latent_x, x, Z_all, patient_id, global_connection_graph_dict, patient_id_to_int)
            mse = loss_tup[0]
            l1 = loss_tup[1]
            contrastive = loss_tup[2]
            loss = loss_tup[3]
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            epoch_mse += mse.item()
            epoch_l1 += l1.item()
            epoch_contrastive += contrastive.item()

        scheduler.step()

        epoch_loss = running_loss / math.ceil(len(dataset)/BATCH_SIZE)
        epoch_mse = epoch_mse / math.ceil(len(dataset)/BATCH_SIZE)
        epoch_l1 = epoch_l1 / math.ceil(len(dataset)/BATCH_SIZE)
        epoch_contrastive = epoch_contrastive / math.ceil(len(dataset)/BATCH_SIZE)
        epoch_time = time.time() - epoch_start_time

        print(f"Epoch [{epoch+1}/{num_epochs}] | Loss: {epoch_loss:.6f} | Epoch time: {epoch_time:.2f}s ")

        log_records.append({
            "epoch": epoch + 1,
            "mse_loss": epoch_mse,
            "l1_loss": epoch_l1,
            "contrastive_loss": epoch_contrastive,
            "total_loss": epoch_loss,
            "epoch_time": epoch_time
        })

        with open(LOG_CSV, 'a') as f:
            f.write(f"{epoch + 1},{epoch_mse:.6f},{epoch_l1:.6f},{epoch_contrastive:.6f},{epoch_loss:.6f},{epoch_time:.2f}\n")

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
    plt.plot(log_df["epoch"], log_df["contrastive_loss"], label="Contrastive Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss Values")
    plt.title("Autoencoder Training Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(LOSS_PLOT, dpi=300)
    plt.close()


if __name__ == "__main__":
    connection_graph_df = pd.read_csv("/home/2023csb1141/CBC_Malignancy_Project/autoencoder/graph_construction/full_unlabeled_data_graphs/cosine_dist_graph.csv")

    global_connection_graph_dict = defaultdict(list)

    for _, row in connection_graph_df.iterrows():
        i = row["source"]
        j = row["target"]
        w = float(row["weight"])

        global_connection_graph_dict[i].append((j, w))
        global_connection_graph_dict[j].append((i, w))  # undirected

    main(global_connection_graph_dict)
