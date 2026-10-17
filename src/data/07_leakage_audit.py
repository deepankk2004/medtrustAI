import pandas as pd
from pathlib import Path
from sklearn.metrics import roc_auc_score

feat = pd.read_parquet("data/cohort/features.parquet")
sp = pd.read_parquet("data/cohort/split.parquet")[["stay_id", "split"]]
df = feat.merge(sp, on="stay_id")
cols = Path("configs/feature_columns.txt").read_text().split("\n")

# 1) column-name screen
banned = ["death", "dod", "expire", "disch", "outtime", "label", "stay_id", "subject", "hadm", "intime", "index_time"]
bad = [c for c in cols if any(k in c.lower() for k in banned)]
print("suspicious column names:", bad or "none")
print("n model features:", len(cols))

# 2) single-feature AUROC screen on TRAIN only (a very high value suggests leakage)
tr = df[df.split == "train"]
res = []
for c in cols:
    x = tr[c]
    if x.nunique() < 2:
        continue
    auc = roc_auc_score(tr.label, x.fillna(x.median()))
    res.append((c, round(auc, 3), round(max(auc, 1 - auc), 3)))
res = pd.DataFrame(res, columns=["feature", "auroc", "strength"]).sort_values("strength", ascending=False)
print("\nTop 12 single-feature AUROC (train):")
print(res.head(12).to_string(index=False))
print("\nFLAG (strength > 0.85):", res[res.strength > 0.85].feature.tolist() or "none")
res.to_csv("reports/leakage_audit_single_feature_auroc.csv", index=False)