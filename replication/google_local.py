import argparse
import gzip
import json
import os
import random
import re

import pandas as pd

SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
STAR_TO_TERNARY = {1: "negative", 2: "negative", 3: "neutral",
                   4: "positive", 5: "positive"}


def sentences(text):
    for part in SENT_SPLIT.split(text.replace("\n", " ")):
        part = " ".join(part.split())
        if part:
            yield part


def usable(s, minlen, maxlen):
    if not (minlen <= len(s) <= maxlen):
        return False
    if sum(c.isalpha() for c in s) < 0.6 * len(s):
        return False
    if not s[0].isupper():
        return False
    return True


def harvest(path, per_star, minlen, maxlen, seed, max_reviews):
    rng = random.Random(seed)
    pools = {1: [], 2: [], 3: [], 4: [], 5: []}
    seen_text = set()
    n_reviews = n_kept = 0

    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if max_reviews and n_reviews >= max_reviews:
                break
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            n_reviews += 1
            text = d.get("text")
            rating = d.get("rating")
            if not text or rating not in pools:
                continue
            for s in sentences(text):
                if not usable(s, minlen, maxlen) or s in seen_text:
                    continue
                seen_text.add(s)
                n_kept += 1
                pools[rating].append({
                    "sentence": s,
                    "review_rating": rating,
                    "star_label": STAR_TO_TERNARY[rating],
                    "gmap_id": d.get("gmap_id"),
                    "review_time": d.get("time"),
                })
            if n_reviews % 200000 == 0:
                print(f"    {n_reviews:,} reviews scanned, "
                      f"{n_kept:,} candidate sentences")

    print(f"  scanned {n_reviews:,} reviews -> {n_kept:,} candidate sentences")
    for star in sorted(pools):
        print(f"    {star}-star: {len(pools[star]):,}")

    out = []
    for star in sorted(pools):
        pool = pools[star]
        take = min(per_star, len(pool))
        if take < per_star:
            print(f"  note: only {take:,} available at {star} stars")
        out.extend(rng.sample(pool, take))
    rng.shuffle(out)
    return pd.DataFrame(out)


def main(args):
    if not os.path.exists(args.input):
        raise SystemExit(
            f"  {args.input} not found.\n"
            f"  Download a per-state review file from\n"
            f"  https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/\n"
            f"  and pass it with --input.")

    print(f"Harvesting prompts from {args.input}")
    df = harvest(args.input, args.per_star, args.minlen, args.maxlen,
                 args.seed, args.max_reviews)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n  wrote {args.out}  ({len(df):,} prompt sentences)")
    print("\n  by star rating:")
    print(df.review_rating.value_counts().sort_index().to_string())
    print("\n  by star-derived ternary label:")
    print(df.star_label.value_counts().to_string())
    print("\n  sample:")
    for s in df.sentence.head(5):
        print(f"    {s}")
    print("\n  Next: run a model over these, keep the sentences where the")
    print("  prediction disagrees with the star-derived label, and send")
    print("  those to validation. That mirrors the paper's Round 1 design.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True,
                   help="review-<State>.json.gz from the McAuley Lab release")
    p.add_argument("--out", default="data/google_local_prompts.csv")
    p.add_argument("--per-star", type=int, default=400)
    p.add_argument("--minlen", type=int, default=30)
    p.add_argument("--maxlen", type=int, default=200)
    p.add_argument("--max-reviews", type=int, default=500000,
                   help="stop after this many reviews; 0 for all")
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args())
