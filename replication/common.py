import json
import os

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
DATA_DIR = os.path.join(REPO_ROOT, "dynasent-v1.1")
RESULTS_DIR = os.path.join(HERE, "results")

ALL_CATS = ("positive", "negative", "neutral", "mixed")
TERNARY_CATS = ("positive", "negative", "neutral")

R1 = {
    split: os.path.join(DATA_DIR, f"dynasent-v1.1-round01-yelp-{split}.jsonl")
    for split in ("train", "dev", "test")
}
R2 = {
    split: os.path.join(DATA_DIR, f"dynasent-v1.1-round02-dynabench-{split}.jsonl")
    for split in ("train", "dev", "test")
}
SST_DEV = os.path.join(DATA_DIR, "sst-dev-validated.jsonl")


def load_dataset(*src_filenames, labels=None):
    data = []
    for filename in src_filenames:
        with open(filename) as f:
            for line in f:
                d = json.loads(line)
                if labels is None or d["gold_label"] in labels:
                    data.append(d)
    return data


def xy(dataset, label_field="gold_label"):
    X = [d["sentence"] for d in dataset]
    y = [d[label_field] for d in dataset]
    return X, y


def ensure_results_dir():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    return RESULTS_DIR


def save_table(df, name, float_format="%.1f"):
    ensure_results_dir()
    csv_path = os.path.join(RESULTS_DIR, f"{name}.csv")
    tex_path = os.path.join(RESULTS_DIR, f"{name}.tex")
    df.to_csv(csv_path)
    with open(tex_path, "w") as f:
        f.write(df.to_latex(float_format=float_format))
    print(f"  -> wrote {csv_path}")
    print(f"  -> wrote {tex_path}")
