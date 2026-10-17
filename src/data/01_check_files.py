import duckdb
from pathlib import Path

con = duckdb.connect()
files = sorted(list(Path("data/raw").glob("*.csv")) + list(Path("data/raw").glob("*.csv.gz")))

if not files:
    print("data/raw mein koi csv / csv.gz file nahi mili")

for f in files:
    size_gb = f.stat().st_size / 1e9
    try:
        cols = con.execute(
            f"DESCRIBE SELECT * FROM read_csv_auto('{f.as_posix()}', sample_size=2000)"
        ).df()
        print(f"{f.name}  ({size_gb:.2f} GB)\n   {list(cols.column_name)}\n")
    except Exception as e:
        print(f"{f.name}  ({size_gb:.2f} GB)  ->  ERROR: {str(e)[:150]}\n")

needed = ["patients", "admissions", "icustays", "chartevents", "labevents", "d_items", "d_labitems"]
have = {f.name.split(".")[0] for f in files}
print("MISSING:", [n for n in needed if n not in have] or "none")