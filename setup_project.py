from pathlib import Path

for d in ["configs", "data/raw", "data/processed", "data/cohort",
          "src/data", "src/preprocessing", "src/models", "src/calibration",
          "src/explainability", "src/reliability", "src/routing", "src/evaluation",
          "experiments", "notebooks", "reports", "tests"]:
    Path(d).mkdir(parents=True, exist_ok=True)

Path(".gitignore").write_text("data/\n.venv/\n__pycache__/\n*.parquet\n*.csv\n*.gz\n.ipynb_checkpoints/\n    !reports/*.csv\n" )
print("done")