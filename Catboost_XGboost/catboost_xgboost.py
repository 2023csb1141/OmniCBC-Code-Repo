"""
CatBoost and XGBoost baselines for CBC diagnosis classification.
Same train/test split, same DROP_COLS, and same label-cleaning as the rest
of the pipeline (TabPFN, TabFM, TabICL, TabLLM). Filtered to the same
10-class valid_classes subset used in the TabPFN/TabLLM comparison runs.
GPU-accelerated where each library supports it.
"""

import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_sample_weight
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
TRAIN_PATH = ""
TEST_PATH = ""
OUTPUT_DIR = Path("")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_COL = "Revised_final_diagnosis"

DROP_COLS = [
    "pseudo_patient_id", "Analyzer_model", "Analyzer_id",
    "RDW-SD(fL)", "RDW-CV(%)", "PDW(fL)", "MPV(fL)", "P-LCR(%)", "PCT(%)",
    "Revised_final_diagnosis",
]

# Same 10-class restriction used in the TabPFN/TabLLM comparison runs.
VALID_CLASSES = [
    'Acute lymphoblastic leukemia', 'Chronic myeloid leukemia', 'Acute myeloid leukemia non M3',
    'Multiple myeloma', 'Aplastic anemia', 'Chronic lymphocytic leukemia',
    'Acute promyelocytic leukemia', 'Acute leukemia', 'Eosinophilia', 'Primary myelofibrosis',
]


def clean_labels(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
    # return s.replace(LABEL_FIXES)
    return s


# ---------------------------------------------------------------------------
# Load + clean + filter to valid_classes
# ---------------------------------------------------------------------------
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

train_df[TARGET_COL] = clean_labels(train_df[TARGET_COL])
test_df[TARGET_COL] = clean_labels(test_df[TARGET_COL])

n_train_before, n_test_before = len(train_df), len(test_df)
train_df = train_df[train_df[TARGET_COL].isin(VALID_CLASSES)].reset_index(drop=True)
test_df = test_df[test_df[TARGET_COL].isin(VALID_CLASSES)].reset_index(drop=True)
print(f"Train: kept {len(train_df)}/{n_train_before} rows after filtering to valid_classes")
print(f"Test:  kept {len(test_df)}/{n_test_before} rows after filtering to valid_classes")

feature_cols = [c for c in train_df.columns if c not in DROP_COLS]

missing_in_test = [c for c in feature_cols if c not in test_df.columns]
if missing_in_test:
    raise ValueError(f"Test set is missing columns present in train: {missing_in_test}")

X_train_raw, X_test_raw = train_df[feature_cols], test_df[feature_cols]

label_encoder = LabelEncoder()
label_encoder.fit(sorted(VALID_CLASSES))  # fixed class set, not whatever survives filtering
y_train = label_encoder.transform(train_df[TARGET_COL])

y_test_str = test_df[TARGET_COL]
known_mask = y_test_str.isin(label_encoder.classes_)
if not known_mask.all():
    dropped = sorted(set(y_test_str[~known_mask]))
    print(f"[warn] Dropping {(~known_mask).sum()} test rows with unseen labels: {dropped}")
X_test_raw = X_test_raw[known_mask]
y_test = label_encoder.transform(y_test_str[known_mask])
target_names = label_encoder.classes_

cat_cols = [c for c in feature_cols if X_train_raw[c].dtype == "object" or str(X_train_raw[c].dtype) == "category"]
print(f"Train: {X_train_raw.shape}, Test: {X_test_raw.shape}, "
      f"Classes: {len(target_names)}, Categorical cols: {cat_cols if cat_cols else 'none (all numeric CBC values)'}")

sample_weight_train = compute_sample_weight(class_weight="balanced", y=y_train)

# ---------------------------------------------------------------------------
# Result collection
# ---------------------------------------------------------------------------
summary_rows = []


def evaluate(name, y_true, y_pred):
    report = classification_report(
        y_true, y_pred, labels=range(len(target_names)),
        target_names=target_names, output_dict=True, zero_division=0
    )
    pd.DataFrame(report).transpose().to_csv(OUTPUT_DIR / f"{name}_report.csv")

    preds_df = pd.DataFrame({
        "true_label": label_encoder.inverse_transform(y_true),
        "predicted_label": label_encoder.inverse_transform(y_pred),
    })
    preds_df.to_csv(OUTPUT_DIR / f"{name}_predictions.csv", index=False)

    summary_rows.append({
        "model": name,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
    })
    print(f"[{name}] accuracy={summary_rows[-1]['accuracy']:.3f} "
          f"macro_f1={summary_rows[-1]['macro_f1']:.3f} "
          f"weighted_f1={summary_rows[-1]['weighted_f1']:.3f}")


# ---------------------------------------------------------------------------
# XGBoost -- GPU via device="cuda" (current API; tree_method stays "hist")
# ---------------------------------------------------------------------------
from xgboost import XGBClassifier
import re

def sanitize_colname(name: str) -> str:
    # XGBoost forbids [, ], and < in feature names -- strip/replace them
    return re.sub(r"[\[\]<]", "", name)

xgb_col_map = {c: sanitize_colname(c) for c in feature_cols}
if len(set(xgb_col_map.values())) != len(xgb_col_map):
    raise ValueError("Sanitizing column names for XGBoost produced duplicate names -- "
                      "check xgb_col_map for collisions before proceeding.")

X_train_xgb = X_train_raw.rename(columns=xgb_col_map)
X_test_xgb = X_test_raw.rename(columns=xgb_col_map)

if cat_cols:
    for c in cat_cols:
        X_train_xgb[xgb_col_map[c]] = X_train_xgb[xgb_col_map[c]].astype("category")
        X_test_xgb[xgb_col_map[c]] = X_test_xgb[xgb_col_map[c]].astype("category")

xgb = XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective="multi:softprob",
    num_class=len(target_names),
    eval_metric="mlogloss",
    tree_method="hist",
    device="cuda",   # GPU acceleration -- falls back with a warning if no GPU visible
    enable_categorical=bool(cat_cols),
    random_state=42,
)
xgb.fit(X_train_xgb, y_train, sample_weight=sample_weight_train)
preds = xgb.predict(X_test_xgb)
evaluate("XGBoost", y_test, preds)

# ---------------------------------------------------------------------------
# CatBoost -- GPU via task_type="GPU"
# ---------------------------------------------------------------------------
from catboost import CatBoostClassifier

cat_idx = [X_train_raw.columns.get_loc(c) for c in cat_cols]

cb = CatBoostClassifier(
    loss_function="MultiClass",
    iterations=500,
    verbose=False,
    auto_class_weights="Balanced",
    random_state=42,
    task_type="GPU",
    devices="0",
)
cb.fit(X_train_raw, y_train, cat_features=cat_idx)
preds = cb.predict(X_test_raw).flatten().astype(int)
evaluate("CatBoost", y_test, preds)

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
summary_df = pd.DataFrame(summary_rows).sort_values("macro_f1", ascending=False)
summary_df.to_csv(OUTPUT_DIR / "catboost_xgboost_summary.csv", index=False)
print("\n=== Summary (sorted by macro F1) ===")
print(summary_df.to_string(index=False))
print(f"\nPer-model reports, predictions, and summary saved to {OUTPUT_DIR}")