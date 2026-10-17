import json, time, joblib, warnings
import numpy as np, pandas as pd, optuna
from pathlib import Path
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
SEED, NJ = 42, 4                       # NJ = CPU threads (apne laptop ke hisaab se)
N_TRIALS = {"LR": 12, "RF": 10, "XGBoost": 25, "LightGBM": 25, "CatBoost": 15}   # slow ho to kam karo
Path("data/models").mkdir(parents=True, exist_ok=True)

feat = pd.read_parquet("data/cohort/features.parquet")
sp = pd.read_parquet("data/cohort/split.parquet")[["stay_id", "split"]]
df = feat.merge(sp, on="stay_id")
df = df[df.split.isin(["train", "calib", "val"])]            # TEST rows dropped here, never used
cols = Path("configs/feature_columns.txt").read_text().split("\n")
tr = df[df.split == "train"]
Xtr, ytr = tr[cols], tr.label
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)

def suggest(name, t):
    if name == "LR":
        return {"C": t.suggest_float("C", 1e-3, 10, log=True)}
    if name == "RF":
        return {"n_estimators": t.suggest_int("n_estimators", 200, 500, step=100),
                "max_depth": t.suggest_int("max_depth", 4, 16),
                "min_samples_leaf": t.suggest_int("min_samples_leaf", 5, 50),
                "max_features": t.suggest_categorical("max_features", ["sqrt", 0.3, 0.5])}
    if name == "XGBoost":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 600),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "max_depth": t.suggest_int("max_depth", 3, 7),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "min_child_weight": t.suggest_int("min_child_weight", 1, 20),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if name == "LightGBM":
        return {"n_estimators": t.suggest_int("n_estimators", 100, 600),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "num_leaves": t.suggest_int("num_leaves", 8, 64),
                "min_child_samples": t.suggest_int("min_child_samples", 20, 100),
                "subsample": t.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": t.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_lambda": t.suggest_float("reg_lambda", 1e-3, 10, log=True)}
    if name == "CatBoost":
        return {"iterations": t.suggest_int("iterations", 200, 600),
                "learning_rate": t.suggest_float("learning_rate", 0.02, 0.15, log=True),
                "depth": t.suggest_int("depth", 3, 7),
                "l2_leaf_reg": t.suggest_float("l2_leaf_reg", 1, 10, log=True)}

def make(name, p):
    if name == "LR":
        return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(max_iter=3000, **p))
    if name == "RF":
        return make_pipeline(SimpleImputer(strategy="median"),
                             RandomForestClassifier(n_jobs=NJ, random_state=SEED, **p))
    if name == "XGBoost":
        return xgb.XGBClassifier(tree_method="hist", eval_metric="logloss", n_jobs=NJ, random_state=SEED, **p)
    if name == "LightGBM":
        return lgb.LGBMClassifier(subsample_freq=1, n_jobs=NJ, random_state=SEED, verbose=-1, **p)
    if name == "CatBoost":
        return CatBoostClassifier(thread_count=NJ, random_seed=SEED, verbose=0, **p)

ev = df[df.split.isin(["calib", "val"])].copy()
preds = ev[["stay_id", "split", "label"]].reset_index(drop=True)
best_params, rows = {}, []

for name in N_TRIALS:
    t0 = time.time()
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED))
    # objective = CV log-loss (proper scoring rule; calibrated probabilities feed the routing)
    study.optimize(lambda t: cross_val_score(make(name, suggest(name, t)), Xtr, ytr,
                                             cv=cv, scoring="neg_log_loss").mean(),
                   n_trials=N_TRIALS[name])
    best_params[name] = study.best_params
    model = make(name, study.best_params).fit(Xtr, ytr)
    joblib.dump(model, f"data/models/{name}.joblib")
    preds[f"p_{name}"] = model.predict_proba(ev[cols])[:, 1]
    v = preds[preds.split == "val"]
    rows.append((name, roc_auc_score(v.label, v[f"p_{name}"]), average_precision_score(v.label, v[f"p_{name}"]),
                 brier_score_loss(v.label, v[f"p_{name}"]), (time.time() - t0) / 60))
    print(f"{name:9s} done | val AUROC={rows[-1][1]:.4f} AUPRC={rows[-1][2]:.4f} Brier={rows[-1][3]:.4f} | {rows[-1][4]:.1f} min")

preds.to_parquet("data/cohort/preds_raw.parquet", index=False)
json.dump(best_params, open("configs/model_params.json", "w"), indent=2)
print("\nSaved: data/cohort/preds_raw.parquet, data/models/*.joblib, configs/model_params.json")