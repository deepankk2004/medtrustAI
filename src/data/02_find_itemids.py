import duckdb
import pandas as pd
from pathlib import Path

pd.set_option("display.max_rows", 500, "display.width", 220, "display.max_colwidth", 60)
Path("reports").mkdir(exist_ok=True)

def find(name):
    for ext in (".csv", ".csv.gz"):
        p = Path("data/raw") / f"{name}{ext}"
        if p.exists():
            return p.as_posix()
    raise FileNotFoundError(f"{name} not found in data/raw")

con = duckdb.connect()

# ---------- Vitals (ICU chartevents) ----------
vitals = con.execute(f"""
    SELECT itemid, label, abbreviation, category, unitname
    FROM read_csv_auto('{find("d_items")}')
    WHERE linksto = 'chartevents' AND (
         lower(label) LIKE '%heart rate%'
      OR lower(label) LIKE '%respiratory rate%'
      OR lower(label) LIKE '%o2 saturation%' OR lower(label) LIKE '%spo2%'
      OR lower(label) LIKE '%temperature%'
      OR lower(label) LIKE '%blood pressure systolic%'
      OR lower(label) LIKE '%blood pressure diastolic%'
      OR lower(label) LIKE '%arterial blood pressure systolic%'
      OR lower(label) LIKE '%arterial blood pressure diastolic%'
      OR lower(label) LIKE '%non invasive blood pressure%'
    )
    ORDER BY label
""").df()
print("=== VITALS ===")
print(vitals.to_string(index=False))

# ---------- Labs (hospital labevents) ----------
labs = con.execute(f"""
    SELECT itemid, label, fluid, category
    FROM read_csv_auto('{find("d_labitems")}')
    WHERE fluid = 'Blood' AND (
         lower(label) LIKE '%creatinine%'
      OR lower(label) LIKE '%white blood%'
      OR lower(label) LIKE '%lactate%'
      OR lower(label) = 'sodium'
      OR lower(label) = 'potassium'
      OR lower(label) LIKE '%hemoglobin%'
      OR lower(label) LIKE '%platelet%'
      OR lower(label) = 'glucose'
    )
    ORDER BY label
""").df()
print("\n=== LABS ===")
print(labs.to_string(index=False))

vitals.to_csv("reports/vitals_itemid_candidates.csv", index=False)
labs.to_csv("reports/labs_itemid_candidates.csv", index=False)
print("\nSaved to reports/")