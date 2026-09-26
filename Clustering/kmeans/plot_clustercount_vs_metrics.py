import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# SECTION 2: Making plots from existing CSV
file_path = "/home/nitin1/MICCAI_CBC/kmeans/kmeans_metrics_vs_numclusters_encoded_cluster9.csv"
save_img_path = "/home/nitin1/MICCAI_CBC/kmeans/kmeans_metrics_vs_numclusters_encoded_cluster9"

# ---------------- Load CSV ----------------
df = pd.read_csv(file_path)
df = df.iloc[:29]
k = df["k"]

# ---------------- Normalization helper ----------------
def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min())

# ---------------- Normalize metrics ----------------
# Lower is better → invert after normalization
inertia_norm =  minmax_norm(df["Inertia_SSE"])
dbi_norm =  minmax_norm(df["Davies_Bouldin_Index"])

# Higher is better → keep as is
# silhouette_norm = minmax_norm(df["Silhouette_Score"])
silhouette_norm = df["Silhouette_Score"]
ch_norm = minmax_norm(df["Calinski_Harabasz_Score"])

# ---------------- Plot ----------------
plt.figure(figsize=(10, 6))

plt.plot(k, inertia_norm, marker="o", label="Inertia")
plt.plot(k, silhouette_norm, marker="o", label="Silhouette")
# plt.plot(k, dbi_norm, marker="o", label="Davies–Bouldin")
plt.plot(k, ch_norm, marker="o", label="Calinski–Harabasz")

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Normalized Scores")
plt.title("Normalized Clustering Metrics vs Number of Clusters")

plt.legend()
plt.grid(True)
plt.tight_layout()

# ---------------- Save ----------------
plt.savefig(f"{save_img_path}_normalized_comparison.png", dpi=300)
# plt.show()


# Ensure k is numeric and evenly spaced
k_vals = k.values.astype(float)
dk = np.mean(np.diff(k_vals))  # assumes roughly uniform spacing

def laplacian(y, dk):
    """
    Discrete second derivative (Laplacian)
    """
    y = np.asarray(y)
    lap = np.zeros_like(y)
    lap[1:-1] = (y[2:] - 2*y[1:-1] + y[:-2]) / (dk**2)
    lap[0] = np.nan
    lap[-1] = np.nan
    return lap

# Compute Laplacians
lap_inertia = laplacian(inertia_norm, dk)
lap_silhouette = laplacian(silhouette_norm, dk)
lap_ch = laplacian(ch_norm, dk)

plt.figure(figsize=(10, 6))

plt.plot(k, lap_inertia, marker="o", label="Laplacian: Inertia")
plt.plot(k, lap_silhouette, marker="o", label="Laplacian: Silhouette")
plt.plot(k, lap_ch, marker="o", label="Laplacian: Calinski–Harabasz")

plt.axhline(0, color="black", linestyle="--", linewidth=1)

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Second Derivative (Curvature)")
plt.title("Laplacian (Second Derivative) of Clustering Scores vs k")

plt.legend()
plt.grid(True)
plt.tight_layout()

# ---------------- Save ----------------
plt.savefig(f"{save_img_path}_laplacian_curvature.png", dpi=300)
# plt.show()




# ---------------- Load CSV ----------------
df = pd.read_csv(file_path)
df = df.iloc[:29]
k = df["k"].values

# ---------------- Difference helper ----------------
def successive_diff(x):
    return np.diff(x)

def minmax_norm(x):
    return (x - x.min()) / (x.max() - x.min() + 1e-8)

# ---------------- Compute differences ----------------
# Lower is better → invert sign
delta_inertia = -successive_diff(df["Inertia_SSE"].values)
delta_dbi = -successive_diff(df["Davies_Bouldin_Index"].values)

# Higher is better → keep sign
delta_silhouette = successive_diff(df["Silhouette_Score"].values)
delta_ch = successive_diff(df["Calinski_Harabasz_Score"].values)

# ---------------- Normalize differences ----------------
delta_inertia_n = minmax_norm(delta_inertia)
delta_dbi_n = minmax_norm(delta_dbi)
delta_silhouette_n = minmax_norm(delta_silhouette)
delta_ch_n = minmax_norm(delta_ch)

# ---------------- Plot ----------------
plt.figure(figsize=(10, 6))

plt.plot(k[1:], delta_inertia_n, marker="o", label="Δ Inertia")
plt.plot(k[1:], delta_silhouette_n, marker="o", label="Δ Silhouette")
# plt.plot(k[1:], delta_dbi_n, marker="o", label="Δ Davies–Bouldin")
plt.plot(k[1:], delta_ch_n, marker="o", label="Δ Calinski–Harabasz")

plt.xlabel("Number of Clusters (k)")
plt.ylabel("Normalized Improvement (Δ score)")
plt.title("Normalized Successive Improvements vs Number of Clusters")

plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{save_img_path}_delta_normalized.png", dpi=300)
# plt.show()

