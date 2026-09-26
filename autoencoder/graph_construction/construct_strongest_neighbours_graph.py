import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
import time


# Load CSV
data_csv_path = ""
results_dir = ""

df = pd.read_csv(data_csv_path)
print(f"Dataframe shape: {df.shape}")

main_feature_columns = ['WBC(10^3/uL)', 'RBC(10^6/uL)', 'HGB(g/dL)', 'HCT(%)', 'MCV(fL)', 'MCH(pg)', 'MCHC(g/dL)', 'PLT(10^3/uL)', 'NRBC#(10^3/uL)', 'NRBC%(%)', 'NEUT#(10^3/uL)', 'LYMPH#(10^3/uL)', 'MONO#(10^3/uL)', 'EO#(10^3/uL)', 'BASO#(10^3/uL)', 'NEUT%(%)', 'LYMPH%(%)', 'MONO%(%)', 'EO%(%)', 'BASO%(%)', 'IG#(10^3/uL)', 'IG%(%)', '[PLT-I(10^3/uL)]', 'MicroR(%)', 'MacroR(%)', '[TNC(10^3/uL)]', '[WBC-N(10^3/uL)]', '[TNC-N(10^3/uL)]', '[BA-N#(10^3/uL)]', '[BA-N%(%)]', '[WBC-D(10^3/uL)]', '[TNC-D(10^3/uL)]', '[NEUT#&(10^3/uL)]', '[NEUT%&(%)]', '[LYMP#&(10^3/uL)]', '[LYMP%&(%)]', '[HFLC#(10^3/uL)]', '[HFLC%(%)]', '[BA-D#(10^3/uL)]', '[BA-D%(%)]', '[NE-SSC(ch)]', '[NE-SFL(ch)]', '[NE-FSC(ch)]', '[LY-X(ch)]', '[LY-Y(ch)]', '[LY-Z(ch)]', '[MO-X(ch)]', '[MO-Y(ch)]', '[MO-Z(ch)]', '[NE-WX]', '[NE-WY]', '[NE-WZ]', '[LY-WX]', '[LY-WY]', '[LY-WZ]', '[MO-WX]', '[MO-WY]', '[MO-WZ]']
patient_id_column = ['pseudo_patient_id']

df = df[patient_id_column + main_feature_columns]

scaler = StandardScaler()
df[main_feature_columns] = scaler.fit_transform(df[main_feature_columns])

X = df[main_feature_columns].values
patient_ids = df[patient_id_column].values
patient_ids = patient_ids.squeeze()

print(f"Dataset shape: {X.shape}, Patient IDs shape: {patient_ids.shape}")


print("Starting k-NN search...")
t = time.time()

k = 20
knn = NearestNeighbors(
    n_neighbors=k + 1,        # +1 to remove self-match
    metric="euclidean",
    # metric="cosine",
    algorithm="auto",
    n_jobs=-1
)

knn.fit(X)

distances, indices = knn.kneighbors(X)

print("Done with KNN")
print("Total time in minutes and seconds:", divmod(time.time() - t, 60))


# Remove self-distance (first neighbor is the point itself)
distances = distances[:, 1:]
indices = indices[:, 1:]

# Build neighbor sets for fast membership checking
neighbor_sets = {
    patient_ids[i]: set(patient_ids[indices[i]])
    for i in range(len(patient_ids))
}

print(f"Building undirected edge graph")
edges = []
for i in range(len(patient_ids)):
    src = patient_ids[i]
    for j, d in zip(indices[i], distances[i]):
        tgt = patient_ids[j]

        # Mutual k-NN condition
        if src in neighbor_sets[tgt]:
            edges.append((src, tgt, float(d)))
edges = list({
    (min(u, v), max(u, v), w)
    for u, v, w in edges
})
edges_df = pd.DataFrame(
    edges,
    columns=["source", "target", "weight"]
)

edges_df.to_csv(os.path.join(results_dir, "euclidean_dist_graph.csv"), index=False)

print("Saved mutual k-NN graph with", len(edges_df), "edges")
