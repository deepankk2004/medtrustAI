import hashlib
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

SEED = 42
FRAC = {"train": 0.50, "calib": 0.15, "val": 0.15, "test": 0.20}

df = pd.read_parquet("data/cohort/features.parquet",
                     columns=["stay_id", "subject_id", "label", "anchor_year_group"])
assert df.subject_id.is_unique and df.stay_id.is_unique   # one row per patient -> patient-level split

def split(d, test_size):
    a, b = train_test_split(d, test_size=test_size, stratify=d.label, random_state=SEED)
    return a, b

rest, test = split(df, FRAC["test"])
rest2, calib = split(rest, FRAC["calib"] / (1 - FRAC["test"]))
train, val = split(rest2, FRAC["val"] / (1 - FRAC["test"] - FRAC["calib"]))

out = pd.concat([train.assign(split="train"), calib.assign(split="calib"),
                 val.assign(split="val"), test.assign(split="test")]).sort_values("stay_id")
out.to_parquet("data/cohort/split.parquet", index=False)

h = hashlib.md5("\n".join(f"{s}:{p}" for s, p in zip(out.stay_id, out.split)).encode()).hexdigest()
Path("configs/split_info.txt").write_text(f"seed={SEED}\nfractions={FRAC}\nmd5={h}\n")

g = out.groupby("split").label.agg(n="size", deaths="sum", mortality="mean").round(3)
print(g.loc[["train", "calib", "val", "test"]])
print("\nanchor_year_group share per split:")
print(pd.crosstab(out.split, out.anchor_year_group, normalize="index").round(3))
print("\nsubject overlap between splits:",
      sum(len(set(out[out.split == a].subject_id) & set(out[out.split == b].subject_id))
          for a, b in [("train", "test"), ("train", "val"), ("train", "calib"), ("val", "test"), ("calib", "test")]))
print("md5:", h)