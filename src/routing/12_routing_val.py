import json
import numpy as np, pandas as pd

NOISE = "0.25"
TOL, TARGETS = 0.015, [0.6, 0.7, 0.8, 0.9]
thr = json.load(open("configs/primary_model.json"))["operating_threshold"]
E = pd.read_parquet(f"data/cohort/ers_components_val_noise{NOISE}.parquet")
pp = pd.read_parquet("data/cohort/primary_preds.parquet")
pp = pp[pp.split == "val"][["stay_id", "p_cal"]]
D = E.merge(pp, on="stay_id").reset_index(drop=True)
n = len(D)
pred = (D.p_cal >= thr).astype(int).values
conf = np.where(pred == 1, D.p_cal, 1 - D.p_cal)
err = (pred != D.label.values).astype(int)
miss = ((pred == 0) & (D.label.values == 1)).astype(int)

# ---- reliability scores (higher = more reliable), percentile ranks w.r.t. VALIDATION ECDF ----
base = ["stab", "rank_cons", "sign_cons", "faith_excess"]
ref = {k: np.sort(D[k].values) for k in base + ["margin_sd"]}
ecdf = lambda k, x: np.searchsorted(ref[k], x, side="right") / len(ref[k])
S = {"ers_equal_mean": np.column_stack([ecdf(k, D[k].values) for k in base]).mean(1),
     "pred_instability": 1 - ecdf("margin_sd", D.margin_sd.values),
     "control_random_score": np.random.default_rng(42).random(n)}
np.savez("configs/routing_ref_val.npz", **ref)

qs = np.linspace(0, 0.6, 31)
rows, chosen = [], {}

def summarize(name, target, acc, tc, te, ch=None, eh=None):
    review = defer = np.nan
    if ch is not None:
        review, defer = (ch ^ eh).mean(), (~ch & ~eh).mean()
    rows.append(dict(router=name, target_cov=target, coverage=acc.mean(), selective_risk=err[acc].mean(),
                     missed_death_rate_accepted=miss[acc].mean(), tau_conf=tc, tau_ers=te,
                     review_rate=review, defer_rate=defer))
    chosen[f"{name}@{target}"] = {"tau_conf": float(tc), "tau_ers": None if te is None else float(te)}

for t in TARGETS:
    tc = np.quantile(conf, 1 - t)
    summarize("conf_only", t, conf >= tc, tc, None)
    for name, s in S.items():
        best = None
        for qc in qs:
            c_thr = np.quantile(conf, qc)
            for qe in qs:
                e_thr = np.quantile(s, qe)
                ch, eh = conf >= c_thr, s >= e_thr
                acc = ch & eh
                if acc.sum() == 0 or abs(acc.mean() - t) > TOL:
                    continue
                risk = err[acc].mean()
                if best is None or risk < best[0]:
                    best = (risk, acc, c_thr, e_thr, ch, eh)
        if best is None:
            print(f"no threshold pair within coverage tolerance for {name}@{t}")
            continue
        summarize(name, t, best[1], best[2], best[3], best[4], best[5])

R = pd.DataFrame(rows)
R["risk_vs_conf_only"] = R.selective_risk - R.groupby("target_cov").selective_risk.transform(lambda x: x.iloc[0])
print(f"val: n={n} | overall error rate={err.mean():.3f} | missed deaths={miss.sum()}")
print("(risk_vs_conf_only < 0 means lower selective risk than confidence-only at the same coverage)\n")
print(R.round(4).sort_values(["target_cov", "router"]).to_string(index=False))
R.to_csv("reports/routing_val.csv", index=False)
json.dump({"noise": NOISE, "operating_threshold": float(thr), "tolerance": TOL, "thresholds": chosen},
          open("configs/routing_val.json", "w"), indent=2)
print("\nSaved reports/routing_val.csv, configs/routing_val.json, configs/routing_ref_val.npz")