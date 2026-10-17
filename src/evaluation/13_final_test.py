import sys, json, subprocess, time, joblib, warnings
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

NOISE, B, SEED, EPS = "0.25", 1000, 42, 1e-4
LOCK = Path(f"reports/TEST_RUN_noise{NOISE}.txt")
git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
if "--force" not in sys.argv:
    if git("tag", "--list", "freeze-all") != "freeze-all":
        sys.exit("Tag 'freeze-all' missing: commit and tag the frozen pipeline first.")
    if git("status", "--porcelain", "src", "configs"):
        sys.exit("Uncommitted changes in src/ or configs/ since the freeze. Investigate (crash fixes only; use --force and note it).")
    if LOCK.exists():
        sys.exit(f"Test already evaluated ({LOCK.read_text().strip()}). Re-running = tuning on test.")
t0 = time.time()

prim = json.load(open("configs/primary_model.json"))
T0, pname = prim["operating_threshold"], prim["primary_model"]
RV = json.load(open("configs/routing_val.json"))
assert abs(RV["operating_threshold"] - T0) < 1e-9 and RV["noise"] == NOISE
ref = dict(np.load("configs/routing_ref_val.npz"))
cols = Path("configs/feature_columns.txt").read_text().split("\n")
feat = pd.read_parquet("data/cohort/features.parquet")
sp = pd.read_parquet("data/cohort/split.parquet")[["stay_id", "split"]]
df = feat.merge(sp, on="stay_id")

base = ["stab", "rank_cons", "sign_cons", "faith_excess"]
ecdf = lambda k, x: np.searchsorted(ref[k], x, side="right") / len(ref[k])
def scores(d):
    R = np.column_stack([ecdf(k, d[k].values) for k in base])
    return {"equal_mean": R.mean(1), "perturb_only": R[:, :3].mean(1), "stab": R[:, 0], "rank_cons": R[:, 1],
            "sign_cons": R[:, 2], "faith_excess": R[:, 3], "pred_instability": 1 - ecdf("margin_sd", d.margin_sd.values)}
def decision(p):
    pr = (p >= T0).astype(int)
    return pr, np.where(pr == 1, p, 1 - p)
def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))
def ece(yv, pv, bins=10):
    edges = np.unique(np.quantile(pv, np.linspace(0, 1, bins + 1)))
    idx = np.clip(np.searchsorted(edges, pv, side="right") - 1, 0, max(len(edges) - 2, 0))
    return sum(abs(yv[idx == b].mean() - pv[idx == b].mean()) * (idx == b).mean() for b in np.unique(idx))
def safe_auc(yy, s): return roc_auc_score(yy, s) if 0 < yy.sum() < len(yy) else np.nan
def mean0(a): return a.mean() if len(a) else np.nan

# ---- validation side (only used to fit frozen H2 models and killer thresholds) ----
Ev = pd.read_parquet(f"data/cohort/ers_components_val_noise{NOISE}.parquet")
pv = pd.read_parquet("data/cohort/primary_preds.parquet")
V = Ev.merge(pv[pv.split == "val"][["stay_id", "p_cal"]], on="stay_id")
predv, confv = decision(V.p_cal.values)
errv = (predv != V.label.values).astype(int)
Sv = scores(V)

# ---- TEST ----
Et = pd.read_parquet(f"data/cohort/ers_components_test_noise{NOISE}.parquet").drop(columns=["label"])
Tt = df[df.split == "test"].merge(Et, on="stay_id", validate="1:1").reset_index(drop=True)
assert len(Tt) == (df.split == "test").sum()
n, y = len(Tt), Tt.label.values
model = joblib.load(f"data/models/{pname}.joblib")
kind, cm = joblib.load("data/models/calibrator.joblib")
p_raw = model.predict_proba(Tt[cols])[:, 1]
assert np.abs(logit(p_raw) - Tt.margin0.values).max() < 1e-2, "ERS file rows do not match model margins"
p_cal = np.clip(cm.predict_proba(logit(p_raw).reshape(-1, 1))[:, 1] if kind == "platt" else cm.predict(p_raw), EPS, 1 - EPS)
pred, conf = decision(p_cal)
err = (pred != y).astype(int)
miss = ((pred == 0) & (y == 1)).astype(int)
St = scores(Tt)

mt = {}
for nm in ["LR", "RF", "XGBoost", "LightGBM", "CatBoost"]:
    p = joblib.load(f"data/models/{nm}.joblib").predict_proba(Tt[cols])[:, 1]
    mt[nm] = (roc_auc_score(y, p), average_precision_score(y, p), brier_score_loss(y, p))

# ---- H2 models: fit on VALIDATION (frozen), evaluate on test ----
lc = lambda c: logit(np.clip(c, EPS, 1 - EPS))
mk = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=3000))
q0 = mk().fit(lc(confv).reshape(-1, 1), errv).predict_proba(lc(conf).reshape(-1, 1))[:, 1]
Q1, coef = {}, {}
for k in St:
    m1 = mk().fit(np.column_stack([lc(confv), Sv[k]]), errv)
    Q1[k] = m1.predict_proba(np.column_stack([lc(conf), St[k]]))[:, 1]
    coef[k] = m1[-1].coef_[0][1]

# ---- killer experiment: thresholds frozen from validation ----
QS, KS = (0.5, 0.7, 0.8, 0.9), ["equal_mean", "pred_instability"]
HC, HI = {}, {}
for q in QS:
    ct = np.quantile(confv, 1 - q); HC[q] = conf >= ct
    for k in KS:
        HI[(k, q)] = St[k] >= np.median(Sv[k][confv >= ct])

# ---- routing: thresholds frozen from validation ----
RS = {"ers_equal_mean": St["equal_mean"], "pred_instability": St["pred_instability"],
      "control_random_score": np.random.default_rng(SEED).random(n)}
RS.update({"fixed_" + k: v for k, v in list(RS.items())})
ROUTE = {}
for key, v in RV["thresholds"].items():
    ch = conf >= v["tau_conf"]
    if v["tau_ers"] is None:
        ROUTE[key] = dict(acc=ch, rev=None, dfr=None)
    else:
        eh = RS[key.split("@")[0]] >= v["tau_ers"]
        ROUTE[key] = dict(acc=ch & eh, rev=ch ^ eh, dfr=~ch & ~eh)

def stats(ix):
    r = {}
    yb, pb, eb, mb, cb, db = y[ix], p_cal[ix], err[ix], miss[ix], conf[ix], pred[ix]
    tp, fp, fn = ((db == 1) & (yb == 1)).sum(), ((db == 1) & (yb == 0)).sum(), ((db == 0) & (yb == 1)).sum()
    r.update(auroc=safe_auc(yb, pb), auprc=average_precision_score(yb, pb), brier=brier_score_loss(yb, pb),
             ece=ece(yb, pb), sens=tp / yb.sum(), spec=((db == 0) & (yb == 0)).sum() / (yb == 0).sum(),
             f1=2 * tp / (2 * tp + fp + fn), error_rate=eb.mean(), missed_death_rate=mb.mean())
    a0 = safe_auc(eb, q0[ix]); r["h2_auc_conf_only"] = a0
    for k in Q1:
        a1 = safe_auc(eb, Q1[k][ix]); r[f"h2_auc|{k}"] = a1; r[f"h2_delta|{k}"] = a1 - a0
    for k in KS:
        for q in QS:
            h = HC[q][ix]; hi = HI[(k, q)][ix][h]; e, m_ = eb[h], mb[h]; p = f"hc|{k}|{q}|"
            r[p + "n"] = h.sum(); r[p + "n_low"] = (~hi).sum()
            r[p + "err_low"], r[p + "err_high"] = mean0(e[~hi]), mean0(e[hi])
            r[p + "diff"] = r[p + "err_low"] - r[p + "err_high"]
            r[p + "miss_low"], r[p + "miss_high"] = mean0(m_[~hi]), mean0(m_[hi])
            r[p + "auroc"] = safe_auc(e, -St[k][ix][h])
    cum = np.cumsum(eb[np.argsort(-cb)])
    for key, Rt in ROUTE.items():
        a = Rt["acc"][ix]; c = int(a.sum())
        if c == 0: continue
        p = f"rt|{key}|"
        r[p + "cov"], r[p + "risk"], r[p + "matched"] = c / len(ix), eb[a].mean(), cum[c - 1] / c
        r[p + "delta"], r[p + "miss_acc"] = r[p + "risk"] - r[p + "matched"], mb[a].mean()
        r[p + "review"] = Rt["rev"][ix].mean() if Rt["rev"] is not None else np.nan
        r[p + "defer"] = Rt["dfr"][ix].mean() if Rt["dfr"] is not None else np.nan
    return r

pt = stats(np.arange(n))
rng = np.random.default_rng(SEED)
bt = pd.DataFrame([stats(rng.integers(0, n, n)) for _ in range(B)])
lo, hi_ = bt.quantile(0.025), bt.quantile(0.975)
pd.DataFrame({"point": pd.Series(pt), "ci_lo": lo, "ci_hi": hi_}).to_csv("reports/test_all_metrics.csv")
f = lambda k: f"{pt[k]:.4f} [{lo[k]:.4f}, {hi_[k]:.4f}]"

print(f"TEST: n={n} deaths={int(y.sum())} | decision threshold={T0:.2f} | errors={err.sum()} ({err.mean():.3f}) | missed deaths={miss.sum()}")
print("\n=== 1. All five models on test (raw probabilities; selection was done on validation) ===")
print(pd.DataFrame(mt, index=["AUROC", "AUPRC", "Brier"]).T.round(4).to_string())
print(f"\n=== 2. Primary model: {pname} + {kind} calibration (95% bootstrap CI) ===")
for k in ["auroc", "auprc", "brier", "ece", "sens", "spec", "f1"]:
    print(f"  {k:6s} {f(k)}")
print("\n=== 3. H2 on test: error-detection AUROC beyond confidence (models fit on validation) ===")
print(f"  confidence only: {f('h2_auc_conf_only')}")
print(f"  {'candidate':18s} {'coef(val)':>9s}  delta AUROC [95% CI]")
for k in Q1:
    ok = "  <-- adds info in expected direction" if lo[f'h2_delta|{k}'] > 0 and coef[k] < 0 else ""
    print(f"  {k:18s} {coef[k]:9.3f}  {f('h2_delta|' + k)}{ok}")
print("\n=== 4. Killer experiment: high-confidence subset, LOW vs HIGH ERS (thresholds frozen on validation) ===")
print("  hypothesis: diff = err_low - err_high > 0, AUROC > 0.5")
for k in KS:
    for q in QS:
        p = f"hc|{k}|{q}|"
        print(f"  {k:16s} topconf~{q:.1f} n={int(pt[p + 'n']):5d} (low-ERS n={int(pt[p + 'n_low']):5d}) | "
              f"err low={pt[p + 'err_low']:.3f} high={pt[p + 'err_high']:.3f} | diff {f(p + 'diff')} | "
              f"AUROC {f(p + 'auroc')} | missed-death rate low={pt[p + 'miss_low']:.3f} high={pt[p + 'miss_high']:.3f}")
print("\n=== 5. Routing on test (frozen thresholds); delta = selective risk - conf-only risk at the SAME coverage ===")
print(f"  base error rate (accept everything) = {err.mean():.3f}")
for key in sorted(ROUTE, key=lambda s: (float(s.split('@')[1]), s)):
    p = f"rt|{key}|"
    if p + "cov" not in pt: continue
    print(f"  {key:30s} cov={pt[p + 'cov']:.3f} risk={pt[p + 'risk']:.4f} matched_conf_only={pt[p + 'matched']:.4f} "
          f"delta {f(p + 'delta')} | missed-death rate accepted={pt[p + 'miss_acc']:.4f} "
          f"| review={pt[p + 'review']:.3f} defer={pt[p + 'defer']:.3f}")

print("\n=== 6. Subgroup performance (descriptive) ===")
Tt["age_band"] = pd.cut(Tt.age, [0, 50, 65, 80, 200], labels=["<50", "50-65", "65-80", "80+"], right=False)
rows = []
for g in ["anchor_year_group", "gender", "age_band"]:
    for v, d in Tt.groupby(g, observed=True):
        ii = d.index.values
        rows.append((g, str(v), len(ii), int(y[ii].sum()), safe_auc(y[ii], p_cal[ii]), brier_score_loss(y[ii], p_cal[ii]), err[ii].mean()))
SG = pd.DataFrame(rows, columns=["group", "value", "n", "deaths", "auroc", "brier", "error_rate"])
print(SG.round(3).to_string(index=False))
SG.to_csv("reports/test_subgroups.csv", index=False)

o = np.argsort(-conf); cov_ = np.arange(1, n + 1) / n
fig, ax = plt.subplots(figsize=(5.8, 4.4))
ax.plot(cov_, np.cumsum(err[o]) / np.arange(1, n + 1), label="confidence-only curve")
mrk, seen = {"ers_equal_mean": "o", "pred_instability": "s", "control_random_score": "x"}, set()
for key in ROUTE:
    nm = key.split("@")[0]
    if not nm.startswith("fixed_") or f"rt|{key}|cov" not in pt: continue
    b = nm[len("fixed_"):]
    ax.scatter(pt[f"rt|{key}|cov"], pt[f"rt|{key}|risk"], marker=mrk[b],
               color={"ers_equal_mean": "C1", "pred_instability": "C2", "control_random_score": "C3"}[b],
               label=("fixed gate: " + b) if b not in seen else None); seen.add(b)
ax.set_xlabel("coverage (accepted fraction)"); ax.set_ylabel("selective risk (error among accepted)")
ax.set_title("Risk-coverage on test"); ax.legend(fontsize=7); fig.tight_layout()
fig.savefig("reports/test_risk_coverage.png", dpi=150)

LOCK.write_text(f"{time.ctime()} | HEAD {git('rev-parse', 'HEAD')}\n")
print(f"\nSaved reports/test_all_metrics.csv, test_subgroups.csv, test_risk_coverage.png | {(time.time() - t0) / 60:.1f} min")