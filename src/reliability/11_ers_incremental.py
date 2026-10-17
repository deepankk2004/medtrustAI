import json, sys
import numpy as np, pandas as pd
from scipy.stats import chi2, rankdata, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, log_loss

NOISE = sys.argv[1] if len(sys.argv) > 1 else "0.25"      # use "0.1" for the sensitivity run
SEED, B = 42, 300
thr = json.load(open("configs/primary_model.json"))["operating_threshold"]

E = pd.read_parquet(f"data/cohort/ers_components_val_noise{NOISE}.parquet")
pp = pd.read_parquet("data/cohort/primary_preds.parquet")
pp = pp[pp.split == "val"][["stay_id", "p_cal"]]
D = E.merge(pp, on="stay_id").reset_index(drop=True)
n = len(D)

pred = (D.p_cal >= thr).astype(int)
D["conf"] = np.where(pred == 1, D.p_cal, 1 - D.p_cal)       # P(decision is correct), consistent with threshold
D["err"] = (pred != D.label).astype(int)
y = D.err.values
c = np.clip(D.conf.values, 1e-4, 1 - 1e-4)
lconf = np.log(c / (1 - c))
print(f"noise={NOISE} | n={n} | threshold={thr:.2f} | errors={y.sum()} ({y.mean():.3f}) | "
      f"missed deaths={int(((pred == 0) & (D.label == 1)).sum())}")

# ---- components -> percentile ranks (validation ECDF, saved as frozen reference) ----
base = ["stab", "rank_cons", "sign_cons", "faith_excess"]
ref = {k: np.sort(D[k].values) for k in base}
np.savez(f"configs/ers_reference_val_noise{NOISE}.npz", **ref)
for k in base:
    D[k + "_r"] = np.searchsorted(ref[k], D[k].values, side="right") / n
Rm = D[[k + "_r" for k in base]].values
S = {"equal_mean": Rm.mean(1), "geo_mean": np.exp(np.log(Rm).mean(1)), "min": Rm.min(1),
     "perturb_only": Rm[:, :3].mean(1), "stab": Rm[:, 0], "rank_cons": Rm[:, 1],
     "sign_cons": Rm[:, 2], "faith_excess": Rm[:, 3],
     "BASELINE_pred_instability": rankdata(-D.margin_sd.values) / n}     # high margin_sd = unreliable

# ---- H2: does ERS add information beyond confidence? ----
lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1e6, max_iter=3000))
def oof(X, seed):
    o = np.zeros(n)
    for a, b in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, y):
        o[b] = lr().fit(X[a], y[a]).predict_proba(X[b])[:, 1]
    return o

X0 = lconf.reshape(-1, 1)
o0 = [oof(X0, s) for s in range(3)]
auc0 = np.mean([roc_auc_score(y, o) for o in o0])
bi = np.random.default_rng(SEED).integers(0, n, (B, n))
auc0_b = np.array([roc_auc_score(y[i], o0[0][i]) for i in bi])
ll = lambda p: -log_loss(y, p, normalize=False)
m0 = LogisticRegression(penalty=None, max_iter=5000).fit(X0, y)
ll0 = ll(m0.predict_proba(X0)[:, 1])

rows = []
for name, s in S.items():
    X1 = np.column_stack([lconf, s])
    o1 = [oof(X1, k) for k in range(3)]
    auc1 = np.mean([roc_auc_score(y, o) for o in o1])
    d_b = np.array([roc_auc_score(y[i], o1[0][i]) for i in bi]) - auc0_b
    m1 = LogisticRegression(penalty=None, max_iter=5000).fit(X1, y)
    rows.append(dict(candidate=name, spearman_vs_conf=spearmanr(s, D.conf).statistic,
                     coef_ers=lr().fit(X1, y)[-1].coef_[0][1], auc_conf_only=auc0, auc_conf_plus=auc1,
                     delta=auc1 - auc0, ci_lo=np.percentile(d_b, 2.5), ci_hi=np.percentile(d_b, 97.5),
                     lrt_p=chi2.sf(2 * (ll(m1.predict_proba(X1)[:, 1]) - ll0), 1)))
T = pd.DataFrame(rows)
print("\nH2: incremental value beyond confidence (coef_ers < 0 = expected direction: higher ERS -> fewer errors)")
print(T.round(4).to_string(index=False))

# ---- pre-declared selection rule ----
ok = T[(~T.candidate.str.startswith("BASELINE")) & (T.coef_ers < 0) & (T.ci_lo > 0)]
if len(ok):
    sel = ok.sort_values("delta", ascending=False).iloc[0].candidate
    verdict = f"SELECTED ERS = {sel} (expected direction, delta CI above 0)"
else:
    sel = "equal_mean"
    verdict = "NO candidate adds significant incremental value in the expected direction (negative result on H2); equal_mean kept as default for descriptive analyses"
print("\nDECISION:", verdict)

# ---- high-confidence subgroup (killer-experiment preview on VALIDATION) ----
def auc_ci(e, s, reps=200):
    if e.sum() < 20: return np.nan, np.nan, np.nan
    r, bs = np.random.default_rng(SEED), []
    for _ in range(reps):
        i = r.integers(0, len(e), len(e))
        if e[i].min() != e[i].max(): bs.append(roc_auc_score(e[i], -s[i]))
    return roc_auc_score(e, -s), *np.percentile(bs, [2.5, 97.5])

sub = []
for name in dict.fromkeys([sel, "equal_mean", "BASELINE_pred_instability"]):
    s = S[name]
    for q in (0.5, 0.7, 0.8, 0.9):
        hc = D.conf.values >= np.quantile(D.conf.values, 1 - q)
        e, hi = y[hc], s[hc] >= np.median(s[hc])
        idx = np.random.default_rng(SEED).integers(0, hc.sum(), (B, hc.sum()))
        eb, hb = e[idx], hi[idx]
        diff_b = (eb * ~hb).sum(1) / (~hb).sum(1) - (eb * hb).sum(1) / hb.sum(1)
        a, alo, ahi = auc_ci(e, s[hc])
        sub.append(dict(ers=name, top_conf_share=q, n=int(hc.sum()), errors=int(e.sum()),
                        err_lowERS=e[~hi].mean(), err_highERS=e[hi].mean(),
                        diff=e[~hi].mean() - e[hi].mean(), diff_lo=np.percentile(diff_b, 2.5),
                        diff_hi=np.percentile(diff_b, 97.5), auroc=a, auc_lo=alo, auc_hi=ahi))
U = pd.DataFrame(sub)
print("\nHigh-confidence subgroup: error rate low-ERS vs high-ERS half (hypothesis: diff > 0, AUROC > 0.5)")
print(U.round(3).to_string(index=False))

T.to_csv(f"reports/ers_incremental_val_noise{NOISE}.csv", index=False)
U.to_csv(f"reports/ers_highconf_subgroup_val_noise{NOISE}.csv", index=False)
json.dump({"noise": NOISE, "selected": sel, "verdict": verdict},
          open(f"configs/ers_selection_val_noise{NOISE}.json", "w"), indent=2)
print("\nSaved reports/ers_*.csv and configs/ers_selection_val_noise" + NOISE + ".json")