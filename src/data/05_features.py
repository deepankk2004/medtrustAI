import duckdb, yaml
import pandas as pd
from pathlib import Path

P = Path("data/processed")
cfg = yaml.safe_load(open("configs/features.yaml"))
LOOKBACK = cfg.get("lab_lookback_hours", 0)
pq = lambda n: f"read_parquet('{(P / (n + '.parquet')).as_posix()}')"

# itemid -> feature map (+ range, source)
rows = []
for src, key in [("vital", "vitals"), ("lab", "labs")]:
    for feat, d in cfg[key].items():
        for iid in d["itemids"]:
            rows.append((iid, feat, d["range"][0], d["range"][1], src))
fmap = pd.DataFrame(rows, columns=["itemid", "feature", "lo", "hi", "source"])

con = duckdb.connect()
con.execute("SET memory_limit='6GB'")
con.execute("CREATE TABLE fmap AS SELECT * FROM fmap")
con.execute(f"CREATE TABLE cohort AS SELECT * FROM read_parquet('data/cohort/cohort.parquet')")

# ---- vitals: window [intime, index]; both charttime and storetime must be <= index ----
con.execute(f"""
CREATE TABLE vit AS
WITH x AS (
  SELECT c.stay_id, m.feature, m.lo, m.hi,
         CASE WHEN c.itemid = 223761 THEN (c.valuenum - 32) * 5.0 / 9 ELSE c.valuenum END AS val
  FROM {pq('chartevents_vitals')} c
  JOIN cohort co USING (stay_id)
  JOIN fmap m ON c.itemid = m.itemid AND m.source = 'vital'
  WHERE c.charttime >= co.intime AND c.charttime <= co.index_time
    AND COALESCE(c.storetime, c.charttime) <= co.index_time
)
SELECT stay_id, feature, AVG(val) AS mean, MIN(val) AS min, MAX(val) AS max, COUNT(*) AS n
FROM x WHERE val BETWEEN lo AND hi GROUP BY 1, 2
""")

# ---- labs: window [intime - lookback, index]; join on subject_id (hadm_id can be null) ----
con.execute(f"""
CREATE TABLE lab AS
WITH x AS (
  SELECT co.stay_id, m.feature, m.lo, m.hi, l.valuenum AS val
  FROM {pq('labevents_labs')} l
  JOIN cohort co ON l.subject_id = co.subject_id
  JOIN fmap m ON l.itemid = m.itemid AND m.source = 'lab'
  WHERE l.charttime >= co.intime - INTERVAL '{LOOKBACK} hours' AND l.charttime <= co.index_time
    AND COALESCE(l.storetime, l.charttime) <= co.index_time
)
SELECT stay_id, feature, AVG(val) AS mean, MIN(val) AS min, MAX(val) AS max, COUNT(*) AS n
FROM x WHERE val BETWEEN lo AND hi GROUP BY 1, 2
""")

agg = con.execute("SELECT * FROM vit UNION ALL SELECT * FROM lab").df()
coh = con.execute("SELECT * FROM cohort").df()

wide = agg.pivot(index="stay_id", columns="feature", values=["mean", "min", "max", "n"])
wide.columns = [f"{f}_{s}" if s != "n" else f"{f}_n" for s, f in wide.columns]
wide = wide.reset_index()

feats = list(cfg["vitals"]) + list(cfg["labs"])
df = coh.merge(wide, on="stay_id", how="left")
feature_cols = ["age"]
df["age"] = df["age_at_admit"]
for f in feats:
    df[f"{f}_missing"] = (df[f"{f}_n"].fillna(0) == 0).astype(int)
    feature_cols += [f"{f}_mean", f"{f}_min", f"{f}_max", f"{f}_missing"]
    for s in ["mean", "min", "max"]:
        if f"{f}_{s}" not in df: df[f"{f}_{s}"] = float("nan")

# Only these columns are model inputs. Everything else is metadata/label.
meta_cols = ["stay_id", "subject_id", "hadm_id", "label", "gender", "first_careunit",
             "anchor_year_group", "intime", "index_time"]
out = df[meta_cols + feature_cols + [f"{f}_n" for f in feats]]
out.to_parquet("data/cohort/features.parquet", index=False)
Path("configs/feature_columns.txt").write_text("\n".join(feature_cols))

# ---- sanity report (aggregates only) ----
print(f"rows={len(out):,}  deaths={int(out.label.sum()):,}  model features={len(feature_cols)}")
rep = pd.DataFrame({
    "missing_%": [100 * out[f"{f}_missing"].mean() for f in feats],
    "missing_%_survivors": [100 * out.loc[out.label == 0, f"{f}_missing"].mean() for f in feats],
    "missing_%_deaths": [100 * out.loc[out.label == 1, f"{f}_missing"].mean() for f in feats],
    "median_mean": [out[f"{f}_mean"].median() for f in feats],
}, index=feats).round(1)
print(rep.to_string())
print("\nunique stay_id:", out.stay_id.nunique(), "| duplicated rows:", out.duplicated("stay_id").sum())
rep.to_csv("reports/feature_missingness.csv")