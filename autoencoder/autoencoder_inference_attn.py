import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from autoencoder_train_attn import AutoEncoder, EncoderBlock, DecoderBlock, SelfAttentionBlock


# -------------------------
# Encoding
# -------------------------
def main():

    SAVE_DIR = ""
    CKPT_NAME = "best.pth"
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
    df1 = pd.read_csv("/home/2023csb1141/CBC_Malignancy_Project/Processed Datasets/unlabeled/combined_sysmex_unlabeled_data_bothcentre.csv")
    print(f"Dataframe shape: {df1.shape}")
    df2 = pd.read_csv("/home/2023csb1141/CBC_Malignancy_Project/Processed Datasets/labeled/PGI_Chd/combined_labeled_sysmex.csv")
    print(f"Dataframe shape: {df2.shape}")

    df1['Revised_final_diagnosis'] = np.nan 
    df2['Hospital'] = 'PGI_Chd'

    main_feature_columns = ['WBC(10^3/uL)', 'RBC(10^6/uL)', 'HGB(g/dL)', 'HCT(%)', 'MCV(fL)', 'MCH(pg)', 'MCHC(g/dL)', 'PLT(10^3/uL)', 'NRBC#(10^3/uL)', 'NRBC%(%)', 'NEUT#(10^3/uL)', 'LYMPH#(10^3/uL)', 'MONO#(10^3/uL)', 'EO#(10^3/uL)', 'BASO#(10^3/uL)', 'NEUT%(%)', 'LYMPH%(%)', 'MONO%(%)', 'EO%(%)', 'BASO%(%)', 'IG#(10^3/uL)', 'IG%(%)', '[PLT-I(10^3/uL)]', 'MicroR(%)', 'MacroR(%)', '[TNC(10^3/uL)]', '[WBC-N(10^3/uL)]', '[TNC-N(10^3/uL)]', '[BA-N#(10^3/uL)]', '[BA-N%(%)]', '[WBC-D(10^3/uL)]', '[TNC-D(10^3/uL)]', '[NEUT#&(10^3/uL)]', '[NEUT%&(%)]', '[LYMP#&(10^3/uL)]', '[LYMP%&(%)]', '[HFLC#(10^3/uL)]', '[HFLC%(%)]', '[BA-D#(10^3/uL)]', '[BA-D%(%)]', '[NE-SSC(ch)]', '[NE-SFL(ch)]', '[NE-FSC(ch)]', '[LY-X(ch)]', '[LY-Y(ch)]', '[LY-Z(ch)]', '[MO-X(ch)]', '[MO-Y(ch)]', '[MO-Z(ch)]', '[NE-WX]', '[NE-WY]', '[NE-WZ]', '[LY-WX]', '[LY-WY]', '[LY-WZ]', '[MO-WX]', '[MO-WY]', '[MO-WZ]']
    patient_id_column = ['pseudo_patient_id', 'Analyzer_model', 'Hospital', 'Revised_final_diagnosis']

    df1 = df1[patient_id_column + main_feature_columns]
    df2 = df2[patient_id_column + main_feature_columns]

    df = pd.concat([df1, df2], ignore_index=True)
    # df = df1.copy()

    # Apply the same scaling to the new data using the saved scaler from the checkpoint
    df[main_feature_columns] = scaler.transform(df[main_feature_columns])

    X = df[main_feature_columns].values
    patient_ids = df[patient_id_column].values

    print(f"Dataset shape: {X.shape}, Patient IDs shape: {patient_ids.shape}")

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)

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

    # also calculate the MPAE mean absolute percentage error
    mape = torch.mean(
        torch.abs((X_tensor - reconstructed) / (torch.abs(X_tensor) + eps))
    ).item()


    print(f"Reconstruction MSE (should be low)        : {mse:.6f}")
    print(f"Normalized Absolute Error (should be low) : {nae:.6f}")
    print(f"Cosine Similarity (should be high)         : {cos_sim:.6f}")
    print(f"Mean Absolute Percentage Error (should be low) : {mape:.6f}")

    print(f"{mse:.6f}")
    print(f"{nae:.6f}")
    print(f"{cos_sim:.6f}")
    print(f"{mape:.6f}")

    # with open(os.path.join(SAVE_DIR, "reconstruction_metrics.txt"), "w") as f:
    #     f.write(f"MSE: {mse:.6f}\n")
    #     f.write(f"Normalized Absolute Error [abs( (x-y)/(|x| + |y|) )]: {nae:.6f}\n")
    #     f.write(f"CosineSimilarity: {cos_sim:.6f}\n")

    latent = latent.cpu().numpy()

    # -------- Save as CSV --------
    latent_df = pd.DataFrame(
        latent,
        columns=[f"z_{i}" for i in range(latent.shape[1])]
    )

    latent_df = pd.concat([latent_df, df[patient_id_column]], axis=1)

    if save_encoded_data:
        latent_df.to_csv(OUTPUT_PATH, index=False)
        print("Saved latent embeddings as CSV:", latent_df.shape)


if __name__ == "__main__":
    main()
