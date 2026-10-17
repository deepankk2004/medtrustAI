import time
import duckdb
import yaml
from pathlib import Path

RAW, OUT = Path("data/raw"), Path("data/processed")
OUT.mkdir(parents=True, exist_ok=True)
cfg = yaml.safe_load(open("configs/features.yaml"))

def find(name):
    for ext in (".csv", ".csv.gz"):
        p = RAW / f"{name}{ext}"
        if p.exists():
            return p.as_posix()
    raise FileNotFoundError(f"{name} not found in data/raw")

con = duckdb.connect()
con.execute("SET memory_limit='4GB'")        # apni RAM ke hisaab se badlo (e.g. 12GB for 16GB laptop)
con.execute("SET threads=4")
con.execute(f"SET temp_directory='{(OUT / 'duck_tmp').as_posix()}'")

# ---- small tables ----
for name in ["patients", "admissions", "icustays"]:
    dst = OUT / f"{name}.parquet"
    if dst.exists():
        print(f"skip {dst.name}"); continue
    t = time.time()
    con.execute(f"COPY (SELECT * FROM read_csv_auto('{find(name)}')) TO '{dst.as_posix()}' (FORMAT PARQUET)")
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{dst.as_posix()}')").fetchone()[0]
    print(f"{name}: {n:,} rows ({time.time()-t:.0f}s)")

# ---- big tables, filtered by itemid ----
def convert(src_name, dst_name, cols, types, ids):
    dst = OUT / dst_name
    if dst.exists():
        print(f"skip {dst_name}"); return
    t = time.time()
    typ = ", ".join(f"'{k}': '{v}'" for k, v in types.items())
    sql = f"""
        COPY (
          SELECT {", ".join(cols)}
          FROM read_csv('{find(src_name)}', header=true, types={{{typ}}})
          WHERE itemid IN ({",".join(map(str, ids))}) AND valuenum IS NOT NULL
        ) TO '{dst.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
    """
    con.execute(sql)
    n = con.execute(f"SELECT COUNT(*) FROM read_parquet('{dst.as_posix()}')").fetchone()[0]
    print(f"{dst_name}: {n:,} rows ({(time.time()-t)/60:.1f} min)")

vital_ids = sorted({i for v in cfg["vitals"].values() for i in v["itemids"]})
lab_ids   = sorted({i for v in cfg["labs"].values()   for i in v["itemids"]})

convert("chartevents", "chartevents_vitals.parquet",
        ["subject_id", "hadm_id", "stay_id", "charttime", "storetime", "itemid", "valuenum", "valueuom"],
        {"subject_id": "BIGINT", "hadm_id": "BIGINT", "stay_id": "BIGINT", "itemid": "INTEGER",
         "charttime": "TIMESTAMP", "storetime": "TIMESTAMP", "valuenum": "DOUBLE"},
        vital_ids)

convert("labevents", "labevents_labs.parquet",
        ["subject_id", "hadm_id", "itemid", "charttime", "storetime", "valuenum", "valueuom"],
        {"subject_id": "BIGINT", "hadm_id": "BIGINT", "itemid": "INTEGER",
         "charttime": "TIMESTAMP", "storetime": "TIMESTAMP", "valuenum": "DOUBLE"},
        lab_ids)

# ---- verification: per-itemid counts + labels + value ranges ----
def verify(parq, dict_name, ids, title):
    df = con.execute(f"""
        SELECT p.itemid, d.label, COUNT(*) AS n,
               ROUND(MEDIAN(p.valuenum),1) AS median, ROUND(MIN(p.valuenum),1) AS min,
               ROUND(MAX(p.valuenum),1) AS max
        FROM read_parquet('{(OUT / parq).as_posix()}') p
        LEFT JOIN read_csv_auto('{find(dict_name)}') d ON p.itemid = d.itemid
        GROUP BY p.itemid, d.label ORDER BY p.itemid
    """).df()
    print(f"\n=== {title} ===")
    print(df.to_string(index=False))
    print("itemids with NO data:", sorted(set(ids) - set(df.itemid)) or "none")

verify("chartevents_vitals.parquet", "d_items", vital_ids, "VITALS CHECK")
verify("labevents_labs.parquet", "d_labitems", lab_ids, "LABS CHECK")