"""A/B the Amazon text field: content only vs title + content.

The paper does not state which column of the Amazon Reviews corpus supplies
the sentence. data/amazon_test.csv was built from `content` alone;
data/amazon_test.titlecontent.csv was built from f"{title}. {content}".
Both files have the same rows in the same order, so sampling the same row
indices from each isolates the text field as the only difference.

    python -u replication\\amazon_ab.py --sample 25000
"""

import argparse
import os
import sys

import pandas as pd

from evaluate import Predictor
from agreement import f1_scores, TERNARY_LABELS

PAPER = {
    "model0": {"positive": 89.4, "negative": 86.6, "neutral": 53.7, "macro": 76.6},
    "model1": {"positive": 83.3, "negative": 78.8, "neutral": 55.4, "macro": 72.5},
}


def read(path, idx=None):
    df = pd.read_csv(path)
    scol = next((c for c in ("sentence", "text", "content") if c in df.columns), None)
    gcol = next((c for c in ("gold_label", "label", "ternary", "y") if c in df.columns), None)
    if scol is None or gcol is None:
        sys.exit(f"columns in {path}: {list(df.columns)} - cannot find text/label")
    df = df[[scol, gcol]].rename(columns={scol: "sentence", gcol: "gold"})
    return df.iloc[idx] if idx is not None else df


def main(a):
    content = os.path.join(a.data_dir, f"amazon_{a.split}.csv")
    titled = os.path.join(a.data_dir, f"amazon_{a.split}.titlecontent.csv")
    for p in (content, titled):
        if not os.path.exists(p):
            sys.exit(f"missing {p}")

    base = read(content)
    other = read(titled)
    if len(base) != len(other):
        sys.exit(f"row counts differ: {len(base):,} vs {len(other):,} - "
                 f"the two builds are not aligned, so this comparison is invalid")

    keep = base.gold.isin(TERNARY_LABELS).values & other.gold.isin(TERNARY_LABELS).values
    idx = pd.Series(range(len(base)))[keep]
    if a.sample and a.sample < len(idx):
        idx = idx.sample(a.sample, random_state=8240)
    idx = idx.sort_values().values

    b = base.iloc[idx].reset_index(drop=True)
    o = other.iloc[idx].reset_index(drop=True)
    if not (b.gold == o.gold).all():
        sys.exit("gold labels differ between the two builds on the same rows - "
                 "the files are not row-aligned")

    same = (b.sentence.astype(str) == o.sentence.astype(str)).sum()
    print(f"amazon {a.split}: {len(base):,} rows, scoring {len(b):,}")
    print(f"identical text in {same:,} of {len(b):,} sampled rows "
          f"({100*same/len(b):.1f}% - these are reviews with no title)")
    print(f"mean length  content only {b.sentence.astype(str).str.len().mean():.0f} chars"
          f"   title+content {o.sentence.astype(str).str.len().mean():.0f} chars\n")

    gold = b.gold.tolist()
    rows = []
    for which in a.models.split(","):
        path = os.path.join("models", f"dynasent_{which}.bin")
        if not os.path.exists(path):
            print(f"skipping {which}: {path} not found")
            continue
        pred_model = Predictor(path, batch_size=a.batch_size)
        res = {}
        for name, frame in (("content only", b), ("title + content", o)):
            print(f"  {which}  {name} ...", flush=True)
            pred = pred_model.predict(frame.sentence.astype(str).tolist())
            res[name] = f1_scores(gold, pred, TERNARY_LABELS)

        print(f"\n{which}  (amazon {a.split}, n={len(b):,})")
        print(f"  {'':16}{'content':>10}{'title+cont':>12}{'delta':>9}{'paper':>9}")
        for key, label in (("positive", "Positive"), ("negative", "Negative"),
                           ("neutral", "Neutral"), ("macro", "Macro avg")):
            c = res["content only"][key]
            t = res["title + content"][key]
            print(f"  {label:16}{c:10.1f}{t:12.1f}{t-c:+9.1f}"
                  f"{PAPER[which][key]:9.1f}")
        cg = res["content only"]["neutral_gap"]
        tg = res["title + content"]["neutral_gap"]
        print(f"  {'NEUTRAL GAP':16}{cg:10.1f}{tg:12.1f}{tg-cg:+9.1f}")

        for name in ("content only", "title + content"):
            f = res[name]
            rows.append({
                "model": which, "split": a.split, "text_field": name,
                "n": len(b),
                "positive": round(f["positive"], 1),
                "negative": round(f["negative"], 1),
                "neutral": round(f["neutral"], 1),
                "macro_f1": round(f["macro"], 1),
                "neutral_gap": round(f["neutral_gap"], 1),
                "paper_neutral": PAPER[which]["neutral"],
                "neutral_vs_paper": round(f["neutral"] - PAPER[which]["neutral"], 1),
            })

    if rows and a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        out = pd.DataFrame(rows)
        out.to_csv(a.out, index=False)
        out.to_latex(a.out.replace(".csv", ".tex"), index=False)
        print(f"\nwrote {a.out}")
        print(out.to_string(index=False))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    p.add_argument("--split", default="test", choices=["test", "dev"])
    p.add_argument("--models", default="model0,model1")
    p.add_argument("--sample", type=int, default=25000, help="0 = all rows")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--out", default="replication/results/table05c_amazon_ab.csv")
    main(p.parse_args())
