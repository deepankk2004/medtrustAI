import json, joblib
import numpy as np, pandas as pd
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, f1_score, recall_score
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

P = pd.read_parquet("data/cohort/preds_raw.parquet")
cal, val = P[P.split == "calib"], P[P.split == "val"]
models = [c[2:] for c in P.columns if c.startswith("p_")]
EPS = 1e-4

def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))

def ece(y, p, bins=10):                      # quantile-bin ECE
    y, p = np.asarray(y), np.asarray(p)
    edges = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, p, side="right") - 1, 0, max(len(edges) - 2, 0))
    return sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx))

def fit_cal(kind, p, y):
    if kind == "platt":
        return kind, LogisticRegression(C=1e6, max_iter=1000).fit(logit(p).reshape(-1, 1), y)
    return kind, IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p, y)

def apply(c, p):
    kind, m = c
    q = m.predict_proba(logit(p).reshape(-1, 1))[:, 1] if kind == "platt" else m.predict(p)
    return np.clip(q, EPS, 1 - EPS)

def boot_auc(y, p, n=300, seed=42):
    rng = np.random.default_rng(seed); y, p = np.asarray(y), np.asarray(p)
    vals = [roc_auc_score(y[i], p[i]) for i in rng.integers(0, len(y), (n, len(y)))]
    return np.percentile(vals, [2.5, 97.5])

rows, cals = [], {}
for m in models:
    auc = roc_auc_score(val.label, val[f"p_{m}"]); lo, hi = boot_auc(val.label, val[f"p_{m}"])
    cand = {}
    for kind in ["platt", "isotonic"]:
        c = fit_cal(kind, cal[f"p_{m}"].values, cal.label.values)       # fit on CALIBRATION split only
        pv = apply(c, val[f"p_{m}"].values)
        cand[kind] = (brier_score_loss(val.label, pv), ece(val.label, pv), c)
    kind = min(cand, key=lambda k: (round(cand[k][0], 5), cand[k][1]))   # lower Brier, tie -> lower ECE
    cals[m] = cand[kind][2]
    rows.append(dict(model=m, auroc=auc, ci_lo=lo, ci_hi=hi,
                     auprc=average_precision_score(val.label, val[f"p_{m}"]),
                     brier_raw=brier_score_loss(val.label, val[f"p_{m}"]), ece_raw=ece(val.label, val[f"p_{m}"]),
                     calibrator=kind, brier_cal=cand[kind][0], ece_cal=cand[kind][1]))

R = pd.DataFrame(rows).round(4)
print(R.to_string(index=False))

# ---- predefined selection rule (frozen in PRD) ----
elig = R[R.auroc >= R.auroc.max() - 0.005]
primary = elig.sort_values("brier_cal").iloc[0]
name, kind = primary.model, primary.calibrator
print(f"\nEligible (AUROC within 0.005 of best): {list(elig.model)}")
print(f"PRIMARY MODEL = {name} | calibrator = {kind}")

# ---- primary: calibrated predictions + operating point (chosen on CALIB, reported on VAL) ----
c = cals[name]
cal_p, val_p = apply(c, cal[f"p_{name}"].values), apply(c, val[f"p_{name}"].values)
grid = np.linspace(0.05, 0.6, 56)
thr = grid[int(np.argmax([f1_score(cal.label, cal_p >= t) for t in grid]))]
print(f"operating threshold (max F1 on calib) = {thr:.2f} | val sensitivity={recall_score(val.label, val_p >= thr):.3f} "
      f"specificity={recall_score(1 - val.label, val_p < thr):.3f} F1={f1_score(val.label, val_p >= thr):.3f}")

out = pd.concat([cal.assign(p_raw=cal[f"p_{name}"], p_cal=cal_p), val.assign(p_raw=val[f"p_{name}"], p_cal=val_p)])
out[["stay_id", "split", "label", "p_raw", "p_cal"]].to_parquet("data/cohort/primary_preds.parquet", index=False)
joblib.dump(c, "data/models/calibrator.joblib")
json.dump({"primary_model": name, "calibrator": kind, "operating_threshold": float(thr),
           "rule": "val AUROC within 0.005 of best -> lowest calibrated Brier"},
          open("configs/primary_model.json", "w"), indent=2)
R.to_csv("reports/model_comparison_val.csv", index=False)

# reliability diagram (validation)
fig, ax = plt.subplots(figsize=(4.5, 4.5))
for lab, p in [("raw", val[f"p_{name}"].values), (f"calibrated ({kind})", val_p)]:
    q = pd.qcut(p, 10, duplicates="drop")
    g = pd.DataFrame({"p": p, "y": val.label.values, "q": q}).groupby("q", observed=True).mean()
    ax.plot(g.p, g.y, marker="o", label=lab)
ax.plot([0, 1], [0, 1], "k--", lw=0.8); ax.set_xlabel("predicted"); ax.set_ylabel("observed")
ax.set_title(f"{name} (validation)"); ax.legend(); fig.tight_layout()
fig.savefig("reports/reliability_primary_val.png", dpi=150)
print("\nSaved: configs/primary_model.json, data/cohort/primary_preds.parquet, reports/model_comparison_val.csv, reports/reliability_primary_val.png")