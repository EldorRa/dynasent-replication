import argparse
import os

import pandas as pd
from sklearn.metrics import classification_report, f1_score

from common import TERNARY_CATS, save_table
from evaluate import Predictor

RECIPES = {
    "tweet_eval": {
        "hf": "cardiffnlp/tweet_eval",
        "config": "sentiment",
        "split": "test",
        "text": "text",
        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
        "neutral_origin": "annotator",
        "note": "SemEval-2017 Task 4A via TweetEval",
    },
    "financial_phrasebank": {
        "hf": "takala/financial_phrasebank",
        "config": "sentences_50agree",
        "split": "train",
        "text": "sentence",
        "label_map": {0: "negative", 1: "neutral", 2: "positive"},
        "neutral_origin": "annotator",
        "note": "Financial PhraseBank, >=50% agreement",
    },
}

REFERENCE_GAPS = {
    "model0": {"SST-3 dev": 39.2, "Yelp dev": 30.4, "Amazon dev": 34.0,
               "Round 2 dev": 21.3},
    "model1": {"SST-3 dev": 43.7, "Yelp dev": 23.1, "Amazon dev": 25.5,
               "Round 1 dev": -2.3},
}


def load_recipe(name, limit=None):
    from external_data import _hf
    r = RECIPES[name]
    ds = _hf(r["hf"], name=r["config"], split=r["split"])
    rows = [(x[r["text"]], r["label_map"][x["label"]]) for x in ds]
    if limit:
        rows = rows[:limit]
    return pd.DataFrame(rows, columns=["sentence", "label"]), r


def load_csv(path, limit=None):
    df = pd.read_csv(path).dropna(subset=["sentence", "label"])
    if limit:
        df = df.head(limit)
    return df, {"note": os.path.basename(path), "neutral_origin": "unknown"}


def neutral_gap(scores):
    return (scores["Positive"] + scores["Negative"]) / 2 - scores["Neutral"]


def score(y_true, y_pred):
    f1 = f1_score(y_true, y_pred, labels=list(TERNARY_CATS), average=None)
    out = {c.title(): s * 100 for c, s in zip(TERNARY_CATS, f1)}
    out["Macro avg"] = sum(out.values()) / 3
    out["Neutral gap"] = neutral_gap(out)
    return out


def main(args):
    if args.dataset:
        df, meta = load_recipe(args.dataset, args.limit)
        name = args.dataset
    else:
        df, meta = load_csv(args.csv, args.limit)
        name = os.path.splitext(os.path.basename(args.csv))[0]

    print(f"{name}: {len(df):,} sentences   ({meta['note']})")
    print(f"  neutral origin: {meta['neutral_origin']}")
    print("  label distribution:")
    for k, v in df.label.value_counts().items():
        print(f"    {k:<9} {v:,}")

    X = df.sentence.astype(str).tolist()
    y = df.label.tolist()

    columns = {}
    for which, path in (("model0", args.model0), ("model1", args.model1)):
        if not path:
            continue
        print(f"\n  {which}: predicting {len(X):,} sentences")
        preds = Predictor(path, batch_size=args.batch_size).predict(X)
        columns[which] = score(y, preds)
        if args.verbose:
            print(classification_report(y, preds, labels=list(TERNARY_CATS),
                                        digits=3, zero_division=0))

    out = pd.DataFrame(columns).reindex(
        ["Positive", "Negative", "Neutral", "Macro avg", "Neutral gap"])
    print(f"\n{name}")
    print(out.round(1).to_string())
    save_table(out.round(1), f"newdata_{name}")

    print("\nNeutral gap in context")
    print("  (mean of positive and negative F1, minus neutral F1)")
    for which in columns:
        print(f"\n  {which}")
        print(f"    {name:<16} {columns[which]['Neutral gap']:>6.1f}   "
              f"<- {meta['neutral_origin']}-assigned neutral")
        for ref, gap in REFERENCE_GAPS[which].items():
            origin = "DynaSent" if "Round" in ref else "star-derived"
            print(f"    {ref:<16} {gap:>6.1f}   <- {origin}")

    print("\n  A gap near the star-derived rows supports the paper's reading;")
    print("  a gap near zero suggests DynaSent's neutral is one definition")
    print("  among several rather than a more coherent one.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dataset", choices=sorted(RECIPES))
    g.add_argument("--csv")
    p.add_argument("--model0", default="models/dynasent_model0.bin")
    p.add_argument("--model1", default="models/dynasent_model1.bin")
    p.add_argument("--limit", type=int)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--verbose", action="store_true")
    main(p.parse_args())
