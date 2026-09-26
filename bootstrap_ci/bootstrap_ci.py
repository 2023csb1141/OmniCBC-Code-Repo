"""
Bootstrap 95% confidence intervals for macro precision / recall / F1 (and SAR)
from prediction CSVs with columns: true_label, predicted_label

Usage: edit FILES below, then  python bootstrap_ci.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

# method -> regime -> CSV path. Leave out any regime you don't have.
FILES = {
    "TabICL": {
        "stratified":  "",
        "prospective": "",
        "cross":       "",
    }
}
REGIMES = ["stratified", "prospective", "cross"]
N_BOOT = 2000
SEED = 0
OUT_CSV = ""


def load(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    y_true = df["Revised_final_diagnosis"].astype(str).str.strip().to_numpy()
    y_pred = df["Predicted_diagnosis"].astype(str).str.strip().to_numpy()
    return y_true, y_pred


def macro_prf(y_true, y_pred):
    # sklearn default: average over classes present in y_true or y_pred
    p, r, f, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    return np.array([p, r, f])


def bootstrap(y_true, y_pred, rng):
    n = len(y_true)
    draws = np.empty((N_BOOT, 3))
    for b in range(N_BOOT):
        idx = rng.integers(0, n, n)
        draws[b] = macro_prf(y_true[idx], y_pred[idx])
    return draws


def fmt(point, lo, hi):
    return f"{point:.2f} [{lo:.2f}, {hi:.2f}]"


rows = []
for method, regimes in FILES.items():
    rng = np.random.default_rng(SEED)
    f1_draws = {}
    row = {"method": method}
    for regime in REGIMES:
        if regime not in regimes:
            continue
        y_true, y_pred = load(regimes[regime])
        point = macro_prf(y_true, y_pred)
        draws = bootstrap(y_true, y_pred, rng)
        lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
        for j, m in enumerate(["P", "R", "F1"]):
            row[f"{regime}_{m}"] = fmt(point[j], lo[j], hi[j])
        row[f"{regime}_n"] = len(y_true)
        row[f"{regime}_classes"] = len(set(y_true))
        f1_draws[regime] = (point[2], draws[:, 2])

    # SAR = (P_prosp - P_cross) / (P_strat - P_cross), bootstrapped by resampling each
    # regime's test set independently (the three test sets are disjoint).
    if all(r in f1_draws for r in REGIMES):
        (s, sd), (p, pd_), (c, cd) = (f1_draws[r] for r in REGIMES)
        sar_point = (p - c) / (s - c) if abs(s - c) > 1e-9 else np.nan
        denom = sd - cd
        with np.errstate(divide="ignore", invalid="ignore"):
            sar_draws = np.where(np.abs(denom) > 1e-9, (pd_ - cd) / denom, np.nan)
        sar_lo, sar_hi = np.nanpercentile(sar_draws, [2.5, 97.5])
        row["SAR"] = fmt(sar_point, sar_lo, sar_hi)
        dt, ds = s - p, p - c
        row["d_temporal"] = fmt(dt, *np.percentile(sd - pd_, [2.5, 97.5]))
        row["d_source"] = fmt(ds, *np.percentile(pd_ - cd, [2.5, 97.5]))
    rows.append(row)

out = pd.DataFrame(rows)
pd.set_option("display.width", 250, "display.max_columns", None)
print(out.to_string(index=False))
out.to_csv(OUT_CSV, index=False)
print(f"\nSaved to {OUT_CSV}  ({N_BOOT} bootstrap resamples, 95% percentile intervals)")