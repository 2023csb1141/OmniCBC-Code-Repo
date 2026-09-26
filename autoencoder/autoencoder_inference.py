import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from autoencoder_train import AutoEncoder, EncoderBlock, DecoderBlock


# -------------------------
# Encoding
# -------------------------
def main():

    CSV_PATH = ""
    SAVE_DIR = ""
    CKPT_NAME = "best_upto_1000.pth"
    save_encoded_data=True
    OUTPUT_PATH = ""

    CKPT_PATH = os.path.join(SAVE_DIR, CKPT_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------- Load checkpoint --------
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)
    scaler = ckpt["scaler"]

    model = AutoEncoder().to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # -------- Load data --------
    data = pd.read_csv(CSV_PATH)
    features = data.iloc[:, :58]
    labels = data["Revised final diagnosis"]
    # features = features[labels.notna()].reset_index(drop=True)

    features = features.reset_index(drop=True)
    X = scaler.transform(features)

    print("Successfully scaled data")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    print("Data tensor shape:", X_tensor.shape)

    # -------- Encode --------
    with torch.no_grad():
        reconstructed, latent = model(X_tensor)

    # -------- Mean Squared Error --------
    mse = nn.MSELoss()(reconstructed, X_tensor).item()

    # -------- Cosine Similarity --------
    cos_sim = torch.nn.functional.cosine_similarity(
        reconstructed.view(reconstructed.size(0), -1),
        X_tensor.view(X_tensor.size(0), -1),
        dim=1
    ).mean().item()

    # -------- Normalized Absolute Error --------
    eps = 1e-8
    nae = torch.mean(
        torch.abs((X_tensor - reconstructed) / (torch.abs(X_tensor) + torch.abs(reconstructed) + eps))
    ).item()

    print(f"Reconstruction MSE (should be low)        : {mse:.6f}")
    print(f"Normalized Absolute Error (should be low) : {nae:.6f}")
    print(f"Cosine Similarity (should be high)         : {cos_sim:.6f}")

    with open(os.path.join(SAVE_DIR, "reconstruction_metrics.txt"), "w") as f:
        f.write(f"MSE: {mse:.6f}\n")
        f.write(f"Normalized Absolute Error [abs( (x-y)/(|x| + |y|) )]: {nae:.6f}\n")
        f.write(f"CosineSimilarity: {cos_sim:.6f}\n")

    latent = latent.cpu().numpy()

    # -------- Save as CSV --------
    latent_df = pd.DataFrame(
        latent,
        columns=[f"z_{i}" for i in range(latent.shape[1])]
    )

    latent_df["Revised final diagnosis"] = labels

    if save_encoded_data:
        latent_df.to_csv(OUTPUT_PATH, index=False)
        print("Saved latent embeddings as CSV:", latent_df.shape)


if __name__ == "__main__":
    main()
