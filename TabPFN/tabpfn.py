import os
import numpy as np
import pandas as pd
import tabpfn_client
from tabpfn_client import TabPFNClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Authenticate: set the token in your shell first ->  export TABPFN_TOKEN=...
os.environ["TABPFN_TOKEN"] = ''
tabpfn_client.set_access_token(os.environ["TABPFN_TOKEN"])

valid_classes = [
    'Acute lymphoblastic leukemia', 'Chronic myeloid leukemia', 'Acute myeloid leukemia non M3',
    'Multiple myeloma', 'Aplastic anemia', 'Chronic lymphocytic leukemia',
    'Acute promyelocytic leukemia', 'Acute leukemia', 'Eosinophilia', 'Primary myelofibrosis',
]

target_col = "Revised_final_diagnosis"
drop_cols = ['pseudo_patient_id', 'Analyzer_model', 'Analyzer_id',
             'RDW-SD(fL)', 'RDW-CV(%)', 'PDW(fL)', 'MPV(fL)', 'P-LCR(%)', 'PCT(%)', 'Revised_final_diagnosis']

TRAIN_CSV = ""
TEST_CSV = ""
PRED_OUT = ""        
N_BOOT, SEED = 2000, 0

# ---------------------------------------------------------------------------
# Load + filter to valid_classes only
# ---------------------------------------------------------------------------
df = pd.read_csv(TRAIN_CSV)
n_before = len(df)
df = df[df[target_col].isin(valid_classes)].reset_index(drop=True)
print(f"Train: kept {len(df)}/{n_before} rows ({n_before - len(df)} dropped for classes outside valid_classes)")
X, y = df.drop(drop_cols, axis=1), df[target_col]

test_df = pd.read_csv(TEST_CSV)
n_before_test = len(test_df)
test_df = test_df[test_df[target_col].isin(valid_classes)].reset_index(drop=True)
print(f"Test: kept {len(test_df)}/{n_before_test} rows ({n_before_test - len(test_df)} dropped for classes outside valid_classes)")
X_test, y_test = test_df.drop(drop_cols, axis=1), test_df[target_col]

# ---------------------------------------------------------------------------
# Train + predict
# ---------------------------------------------------------------------------
model = TabPFNClassifier()
model.fit(X, y)
predictions = np.asarray(model.predict(X_test))

# Save predictions so this run can also go through bootstrap_ci.py with the others
os.makedirs(os.path.dirname(PRED_OUT), exist_ok=True)
pd.DataFrame({"true_label": y_test.to_numpy(), "predicted_label": predictions}).to_csv(PRED_OUT, index=False)
print(f"Saved predictions to {PRED_OUT}")

# ---------------------------------------------------------------------------
# Report + bootstrap 95% confidence intervals
# ---------------------------------------------------------------------------
# Macro averaging over classes present in y_true or y_pred (sklearn default), the same
# convention as bootstrap_ci.py, so all methods in Table 5 are computed identically.
print("\nClassification Report")
print(classification_report(y_test, predictions, zero_division=0))

y_true_arr = y_test.to_numpy()

def macro_prf(t, p):
    pr, rc, f1, _ = precision_recall_fscore_support(t, p, average="macro", zero_division=0)
    return np.array([pr, rc, f1])

point = macro_prf(y_true_arr, predictions)
rng = np.random.default_rng(SEED)
n = len(y_true_arr)
draws = np.empty((N_BOOT, 3))
for b in range(N_BOOT):
    idx = rng.integers(0, n, n)
    draws[b] = macro_prf(y_true_arr[idx], predictions[idx])
lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)

print(f"Bootstrap ({N_BOOT} resamples, n={n}, {len(set(y_true_arr))} classes in test set)")
for j, m in enumerate(["Macro P ", "Macro R ", "Macro F1"]):
    print(f"  {m}: {point[j]:.2f} [{lo[j]:.2f}, {hi[j]:.2f}]")