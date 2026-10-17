import duckdb
from pathlib import Path

P = Path("data/processed")
Path("data/cohort").mkdir(parents=True, exist_ok=True)
Path("reports").mkdir(exist_ok=True)
con = duckdb.connect()
pq = lambda n: f"read_parquet('{(P / (n + '.parquet')).as_posix()}')"

con.execute(f"""
CREATE TABLE base AS
SELECT i.subject_id, i.hadm_id, i.stay_id, i.first_careunit, i.intime, i.outtime,
       a.admittime, a.dischtime, a.deathtime, a.hospital_expire_flag,
       p.gender, p.anchor_age, p.anchor_year, p.anchor_year_group,
       p.anchor_age + (EXTRACT(year FROM a.admittime) - p.anchor_year) AS age_at_admit,
       i.intime + INTERVAL 24 HOUR AS index_time,
       ROW_NUMBER() OVER (PARTITION BY i.subject_id ORDER BY i.intime) AS stay_rank
FROM {pq('icustays')} i
JOIN {pq('admissions')} a USING (subject_id, hadm_id)
JOIN {pq('patients')} p USING (subject_id)
""")

steps = [
    ("All ICU stays (linked to admission)", "TRUE"),
    ("First ICU stay per patient",          "stay_rank = 1"),
    ("Adult (age >= 18)",                   "age_at_admit >= 18"),
    ("ICU stay >= 24h",                     "outtime >= index_time"),
    ("Alive at index (no death <= index)",  "(deathtime IS NULL OR deathtime > index_time)"),
    ("Hospital discharge after index",      "dischtime > index_time"),
    ("Outcome label present",               "hospital_expire_flag IS NOT NULL"),
]

rows, where = [], []
for name, cond in steps:
    where.append(cond)
    n, deaths = con.execute(
        f"SELECT COUNT(*), COALESCE(SUM(hospital_expire_flag),0) FROM base WHERE {' AND '.join(where)}"
    ).fetchone()
    rows.append((name, n, int(deaths)))
    print(f"{name:42s} n={n:>8,}  deaths={int(deaths):>6,}  mortality={deaths/n:.1%}")

import pandas as pd
pd.DataFrame(rows, columns=["step", "n_patients", "n_deaths"]).to_csv("reports/attrition.csv", index=False)

final_where = " AND ".join(where)
con.execute(f"""
COPY (SELECT subject_id, hadm_id, stay_id, first_careunit, intime, index_time, outtime,
             gender, age_at_admit, anchor_year_group,
             CAST(hospital_expire_flag AS INTEGER) AS label
      FROM base WHERE {final_where})
TO 'data/cohort/cohort.parquet' (FORMAT PARQUET)
""")

# sanity checks
print("\nexpire_flag=1 but deathtime missing:",
      con.execute(f"SELECT COUNT(*) FROM base WHERE {final_where} AND hospital_expire_flag=1 AND deathtime IS NULL").fetchone()[0])
print("\nanchor_year_group distribution:")
print(con.execute("SELECT anchor_year_group, COUNT(*) n, ROUND(AVG(label),3) mortality "
                  "FROM read_parquet('data/cohort/cohort.parquet') GROUP BY 1 ORDER BY 1").df())
print("\nSaved: data/cohort/cohort.parquet, reports/attrition.csv")