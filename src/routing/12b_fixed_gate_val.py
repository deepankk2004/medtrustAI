import json
import numpy as np, pandas as pd

NOISE, GATE_Q, TARGETS = "0.25", 0.15, [0.6, 0.7, 0.8]
RV = json.load(open("configs/routing_val.json"))
thr = RV["operating_threshold"]
E = pd.read_parquet(f"data/cohort/ers_components_val_noise{NOISE}.parquet")
pp = pd.read_parquet("data/cohort/primary_preds.parquet")
D = E.merge(pp[pp.split == "val"][["stay_id", "p_cal"]], on="stay_id").reset_index(drop=True)
n = len(D)
pred = (D.p_cal >= thr).astype(int).values
conf = np.where(pred == 1, D.p_cal, 1 - D.p_cal)
err = (pred != D.label.values).astype(int)
miss = ((pred == 0) & (D.label.values == 1)).astype(int)

ref = dict(np.load("configs/routing_ref_val.npz"))
base = ["stab", "rank_cons", "sign_cons", "faith_excess"]
ecdf = lambda k, x: np.searchsorted(ref[k], x, side="right") / len(ref[k])
S = {"ers_equal_mean": np.column_stack([ecdf(k, D[k].values) for k in base]).mean(1),
     "pred_instability": 1 - ecdf("margin_sd", D.margin_sd.values),
     "control_random_score": np.random.default_rng(42).random(n)}

order = np.argsort(-conf)
cum = np.cumsum(err[order])
rows = []
for name, s in S.items():
    e_thr = np.quantile(s, GATE_Q)
    eh = s >= e_thr
    for t in TARGETS:
        k = int(round(t * n))
        if k > eh.sum():
            print(f"skip {name}@{t}: gate leaves only {eh.mean():.2f} of patients"); continue
        c_thr = np.sort(conf[eh])[::-1][k - 1]          # accept the top-k most confident among gate-passing patients
        ch = conf >= c_thr
        acc = ch & eh
        c = int(acc.sum())
        rows.append(dict(router="fixed_" + name, target_cov=t, coverage=c / n, selective_risk=err[acc].mean(),
                         conf_only_matched=cum[c - 1] / c, missed_death_rate_accepted=miss[acc].mean(),
                         review_rate=(ch ^ eh).mean(), defer_rate=(~ch & ~eh).mean(), tau_conf=c_thr, tau_ers=e_thr))
        RV["thresholds"][f"fixed_{name}@{t}"] = {"tau_conf": float(c_thr), "tau_ers": float(e_thr)}

R = pd.DataFrame(rows)
R["delta_vs_conf_only"] = R.selective_risk - R.conf_only_matched
print(f"val: gate = reject lowest {GATE_Q:.0%} of each reliability score; conf threshold set to hit the target coverage")
print("(delta < 0 means lower selective risk than confidence-only at the same coverage)\n")
print(R.round(4).sort_values(["target_cov", "router"]).to_string(index=False))
R.to_csv("reports/routing_val_fixed_gate.csv", index=False)
RV["fixed_gate_quantile"] = GATE_Q
json.dump(RV, open("configs/routing_val.json", "w"), indent=2)
print("\nUpdated configs/routing_val.json with fixed_* thresholds")