import sys, time, json, joblib
import numpy as np, pandas as pd, yaml, shap
from pathlib import Path
from sklearn.metrics import roc_auc_score

split_name = sys.argv[1] if len(sys.argv) > 1 else "val"
n_sub = int(sys.argv[2]) if len(sys.argv) > 2 and sys.argv[2].isdigit() else 0     # 0 = all patients
if split_name == "test" and "--frozen" not in sys.argv:
    sys.exit("TEST is locked. Run on test only after everything is frozen (add --frozen).")

SEED, K, TOPK, R = 42, 20, 5, 10
NOISE = float(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 0.10
cfg = yaml.safe_load(open("configs/features.yaml"))
prim = json.load(open("configs/primary_model.json"))
cols = Path("configs/feature_columns.txt").read_text().split("\n")
model = joblib.load(f"data/models/{prim['primary_model']}.joblib")

feat = pd.read_parquet("data/cohort/features.parquet")
sp = pd.read_parquet("data/cohort/split.parquet")[["stay_id", "split"]]
df = feat.merge(sp, on="stay_id")
tr = df[df.split == "train"]
ev = df[df.split == split_name].reset_index(drop=True)
if n_sub:
    ev = ev.sample(n_sub, random_state=SEED).reset_index(drop=True)
X0 = ev[cols].astype(float).values
n = len(X0)
med = np.nanmedian(tr[cols].astype(float).values, axis=0)
tag = split_name + (f"_sub{n_sub}" if n_sub else "") + f"_noise{NOISE:g}"

# ---- plausible perturbation: same shift on mean/min/max, clipped to physiological range ----
ci = {c: i for i, c in enumerate(cols)}
allv = {**cfg["vitals"], **cfg["labs"]}
groups = []
for v, d in allv.items():
    q = tr[f"{v}_mean"].quantile([0.25, 0.75])
    groups.append(([ci[f"{v}_mean"], ci[f"{v}_min"], ci[f"{v}_max"]], NOISE * (q.iloc[1] - q.iloc[0]), *d["range"]))

def perturb(X, rng):
    Xp = X.copy()
    for idxs, sigma, lo, hi in groups:
        Xp[:, idxs] = np.clip(Xp[:, idxs] + rng.normal(0, sigma, size=(len(X), 1)), lo, hi)   # NaN stays NaN
    return Xp

def margin(X):
    D = pd.DataFrame(X, columns=cols); nm = prim["primary_model"]
    if nm == "XGBoost":  return model.predict(D, output_margin=True)
    if nm == "LightGBM": return model.predict(D, raw_score=True)
    return model.predict(D, prediction_type="RawFormulaVal")

expl = shap.TreeExplainer(model)
def phi(X):
    s = expl.shap_values(pd.DataFrame(X, columns=cols))
    return s[1] if isinstance(s, list) else s

def ranks(a): return np.argsort(np.argsort(a, axis=1), axis=1).astype(float)
def topmask(a):
    m = np.zeros(a.shape, bool); np.put_along_axis(m, np.argsort(-a, axis=1)[:, :TOPK], True, axis=1); return m

# ---- original explanation ----
t0 = time.time()
P0, m0 = phi(X0), margin(X0)
ev0 = np.ravel(expl.expected_value)[0]
print(f"{n} patients | SHAP additivity max error: {np.abs(P0.sum(1) + ev0 - m0).max():.5f}")
A0 = np.abs(P0); idx0 = np.argsort(-A0, axis=1)[:, :TOPK]; mask0 = topmask(A0); r0 = ranks(A0)
r0c = r0 - r0.mean(1, keepdims=True)
sgn0 = np.sign(np.take_along_axis(P0, idx0, 1))

acc = {k: np.zeros(n) for k in ["stab", "jacc", "spear", "sign"]}
ms, rng = [], np.random.default_rng(SEED)
for k in range(K):
    Xk = perturb(X0, rng); Pk = phi(Xk); ms.append(margin(Xk))
    cos = (P0 * Pk).sum(1) / (np.linalg.norm(P0, axis=1) * np.linalg.norm(Pk, axis=1) + 1e-12)
    acc["stab"] += np.clip(cos, 0, 1)
    Ak = np.abs(Pk); inter = (mask0 & topmask(Ak)).sum(1)
    acc["jacc"] += inter / (2 * TOPK - inter)
    rkc = ranks(Ak) - ranks(Ak).mean(1, keepdims=True)
    acc["spear"] += np.clip((r0c * rkc).sum(1) / np.sqrt((r0c ** 2).sum(1) * (rkc ** 2).sum(1) + 1e-12), 0, 1)
    acc["sign"] += (np.sign(np.take_along_axis(Pk, idx0, 1)) == sgn0).mean(1)
    if k == 0 or (k + 1) % 5 == 0:
        print(f"perturbation {k + 1}/{K}  ({(time.time() - t0) / 60:.1f} min elapsed)")

# ---- faithfulness: delete top-k features (-> train median), expect logit to move in SHAP direction ----
def delete(X, idx):
    Xm = X.copy(); Xm[np.arange(len(X))[:, None], idx] = med[idx]; return Xm
sg = np.sign(np.take_along_axis(P0, idx0, 1).sum(1)); sg[sg == 0] = 1
denom = np.take_along_axis(A0, idx0, 1).sum(1) + 1e-9
d_top = (m0 - margin(delete(X0, idx0))) * sg
rr = np.random.default_rng(SEED + 1)
d_rand = np.mean([(m0 - margin(delete(X0, np.argsort(rr.random((n, len(cols))), axis=1)[:, :TOPK]))) * sg
                  for _ in range(R)], axis=0)

out = pd.DataFrame({"stay_id": ev.stay_id.values, "label": ev.label.values, "margin0": m0,
                    "stab": acc["stab"] / K, "rank_jacc": acc["jacc"] / K, "rank_spear": acc["spear"] / K,
                    "sign_cons": acc["sign"] / K, "faith": np.clip(d_top / denom, 0, 1),
                    "faith_delta": d_top, "faith_delta_rand": d_rand, "margin_sd": np.std(ms, axis=0)})
out["rank_cons"] = (out.rank_jacc + out.rank_spear) / 2
out["faith_excess"] = np.clip((d_top - d_rand) / denom, 0, 1)
out.to_parquet(f"data/cohort/ers_components_{tag}.parquet", index=False)
pd.DataFrame(P0, columns=cols).assign(stay_id=ev.stay_id.values).to_parquet(f"data/cohort/shap_{tag}.parquet", index=False)

# ---- summary ----
comps = ["stab", "rank_cons", "sign_cons", "faith", "faith_excess"]
print("\nComponent summary:")
print(out[comps + ["margin_sd"]].describe().loc[["mean", "std", "min", "25%", "50%", "75%", "max"]].round(3).to_string())
print(f"\nFaithfulness check: mean logit change, top-{TOPK} deletion = {d_top.mean():.3f} vs random-{TOPK} deletion = {d_rand.mean():.3f}")

pp = pd.read_parquet("data/cohort/primary_preds.parquet")
pp = pp[pp.stay_id.isin(out.stay_id)]
if len(pp):
    o = out.merge(pp[["stay_id", "p_cal"]], on="stay_id")
    thr = json.load(open("configs/primary_model.json"))["operating_threshold"]
    o["conf"] = np.maximum(o.p_cal, 1 - o.p_cal)
    o["err"] = ((o.p_cal >= thr).astype(int) != o.label).astype(int)
    print(f"\nPreview (error rate={o.err.mean():.3f}, n={len(o)}; noisy if subset):")
    print(f"  {'':10s} {'spearman_vs_conf':>17s} {'AUROC_error_detect':>19s}")
    print(f"  {'conf':10s} {'-':>17s} {roc_auc_score(o.err, -o.conf):>19.3f}")
    for c in comps:
        print(f"  {c:10s} {o[c].corr(o.conf, method='spearman'):>17.3f} {roc_auc_score(o.err, -o[c]):>19.3f}")
    print(f"  {'margin_sd':10s} {o.margin_sd.corr(o.conf, method='spearman'):>17.3f} {roc_auc_score(o.err, o.margin_sd):>19.3f}")
print(f"\nSaved: data/cohort/ers_components_{tag}.parquet, shap_{tag}.parquet | total {(time.time() - t0) / 60:.1f} min")