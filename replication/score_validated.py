import argparse
import os

import pandas as pd

from evaluate import Predictor
from agreement import f1_scores, TERNARY_LABELS

LABELS = TERNARY_LABELS


def report(name, gold, pred):
    f = f1_scores(gold, pred, LABELS)
    print(f"\n{name}")
    print(f"  positive        {f['positive']:.1f}")
    print(f"  negative        {f['negative']:.1f}")
    print(f"  neutral         {f['neutral']:.1f}")
    print(f"  macro F1        {f['macro']:.1f}")
    print(f"  pos & neg mean  {(f['positive']+f['negative'])/2:.1f}")
    print(f"  NEUTRAL GAP     {f['neutral_gap']:.1f}")
    return f


def main(a):
    df = pd.read_csv(a.validated).dropna(subset=["sentence", "gold_label"])
    bad = df[~df.gold_label.isin(LABELS)]
    if len(bad):
        print(f"warning: dropping {len(bad):,} rows whose gold_label is not "
              f"one of {LABELS}: {sorted(bad.gold_label.unique())}")
    df = df[df.gold_label.isin(LABELS)]
    sentences = df.sentence.astype(str).tolist()
    gold = df.gold_label.tolist()
    print(f"{len(df):,} validated sentences from {a.validated}")
    print("gold label distribution: "
          + ", ".join(f"{k} {v:,}"
                      for k, v in df.gold_label.value_counts().items()))

    rows = []
    for label, path in (("Model 0", a.model0), ("Model 1", a.model1)):
        if not os.path.exists(path):
            print(f"\nskipping {label}: {path} not found")
            continue
        pred = Predictor(path, batch_size=a.batch_size).predict(sentences)
        f = report(f"{label}  ({os.path.basename(path)})", gold, pred)
        rows.append({
            "model": label,
            "positive": round(f["positive"], 1),
            "negative": round(f["negative"], 1),
            "pos_neg_mean": round((f["positive"] + f["negative"]) / 2, 1),
            "neutral": round(f["neutral"], 1),
            "macro_f1": round(f["macro"], 1),
            "neutral_gap": round(f["neutral_gap"], 1),
        })

    if "star_label" in df.columns:
        sub = df.dropna(subset=["star_label"])
        sub = sub[sub.star_label.isin(LABELS)]
        if len(sub):
            agree = (sub.star_label == sub.gold_label).sum()
            print(f"\nthe star-derived label against the validated gold")
            print(f"  agrees on    {agree:,} of {len(sub):,} "
                  f"({100*agree/len(sub):.1f}%)")
            print(f"  contradicted {len(sub)-agree:,} of {len(sub):,} "
                  f"({100*(len(sub)-agree)/len(sub):.1f}%)")
            if "review_rating" in sub.columns:
                print("\n  contradiction rate by star rating")
                g = (sub.assign(bad=sub.star_label != sub.gold_label)
                        .groupby("review_rating").bad.agg(["sum", "count"]))
                g["rate_%"] = (100 * g["sum"] / g["count"]).round(1)
                g = g.rename(columns={"sum": "contradicted", "count": "n"})
                print(g.to_string())
                print("\n  NOTE: these are conditional on the sentence having")
                print("  been flagged by Model 1 in the harvest. They are the")
                print("  precision of the harvesting rule, NOT the rate at")
                print("  which star labels are wrong in the population.")

    if rows and a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        out = pd.DataFrame(rows)
        out.to_csv(a.out, index=False)
        out.to_latex(a.out.replace(".csv", ".tex"), index=False)
        print(f"\nwrote {a.out} and {a.out.replace('.csv', '.tex')}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--validated", default="results/validated_round3.csv")
    p.add_argument("--model0", default="models/dynasent_model0.bin")
    p.add_argument("--model1", default="models/dynasent_model1.bin")
    p.add_argument("--out", default="results/table06_new_data.csv")
    p.add_argument("--batch-size", type=int, default=64)
    main(p.parse_args())
