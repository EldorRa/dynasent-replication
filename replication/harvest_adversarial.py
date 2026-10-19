import argparse
import os

import pandas as pd

from evaluate import Predictor

TERNARY = ("positive", "negative", "neutral")


def main(args):
    df = pd.read_csv(args.prompts).dropna(subset=["sentence"])
    print(f"{len(df):,} prompt sentences from {args.prompts}")

    predictor = Predictor(args.model, batch_size=args.batch_size)
    sentences = df.sentence.astype(str).tolist()
    df["model_label"] = predictor.predict(sentences)

    df["adversarial"] = df.model_label != df.star_label
    hits = df[df.adversarial].copy()

    rate = 100 * len(hits) / len(df)
    print(f"\nmodel disagrees with the star-derived label on "
          f"{len(hits):,} of {len(df):,} sentences ({rate:.2f}%)")
    print(f"  paper reports 46.53% for Round 1 and 18.71% for Round 2")

    print("\ndisagreement rate by star rating")
    by_star = df.groupby("review_rating").adversarial.agg(["sum", "count"])
    by_star["rate"] = 100 * by_star["sum"] / by_star["count"]
    print(by_star.to_string())

    print("\nwhat the model says when it disagrees")
    print(pd.crosstab(df[df.adversarial].star_label,
                      df[df.adversarial].model_label).to_string())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    hits.to_csv(args.out, index=False)
    print(f"\n  wrote {args.out}  ({len(hits):,} candidates)")

    extreme = hits[hits.review_rating.isin([1, 5])]
    print(f"  of which {len(extreme):,} come from 1-star or 5-star reviews")
    print("  (the paper sampled these specifically: a 1-star review the")
    print("   model calls positive, or a 5-star review it calls negative)")

    print("\nsample candidates")
    for _, r in hits.head(8).iterrows():
        print(f"  [{r.review_rating} star -> star label {r.star_label};"
              f" model says {r.model_label}]")
        print(f"    {r.sentence}")

    print("\n  Next: these go to validation. Each needs several independent")
    print("  judgements, and the majority becomes the gold label, exactly")
    print("  as in the paper's Section 3.3.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--prompts", default="data/google_local_prompts.csv")
    p.add_argument("--model", default="models/dynasent_model1.bin")
    p.add_argument("--out", default="data/round3_candidates.csv")
    p.add_argument("--batch-size", type=int, default=64)
    main(p.parse_args())
