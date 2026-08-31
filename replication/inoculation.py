import argparse
import os
import random

import pandas as pd

from common import R1, TERNARY_CATS, load_dataset, ensure_results_dir

SIZES = [0, 10, 100, 500, 1000, 3000, 5000, 7000, 9000, 10000]


def sample_round1(n, seed):
    data = load_dataset(R1["train"], labels=TERNARY_CATS)
    random.Random(seed).shuffle(data)
    subset = data[:n]
    return pd.DataFrame({
        "sentence": [d["sentence"] for d in subset],
        "label": [d["gold_label"] for d in subset]})


def plot(df, out_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    keep = ["SST-3 Dev", "Yelp Dev", "Amazon Dev", "Round 1 Dev"]
    styles = {"SST-3 Dev": ("o", "-"), "Yelp Dev": ("^", "--"),
              "Amazon Dev": ("s", "-"), "Round 1 Dev": ("D", "-")}
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, cat in zip(axes, ("Positive", "Negative", "Neutral")):
        sub = df[df.category == cat]
        for split in keep:
            s = sub[sub.split == split].sort_values("n")
            if s.empty:
                continue
            marker, ls = styles[split]
            ax.plot(range(len(s)), s.f1 / 100, marker=marker, ls=ls,
                    label=split)
        ax.set_xticks(range(len(SIZES)))
        ax.set_xticklabels([str(n) for n in SIZES], rotation=45)
        ax.set_title(cat)
        ax.set_xlabel("# of Fine-tuning Examples")
    axes[0].set_ylabel("F1")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    path = os.path.join(out_dir, "figure03_inoculation.png")
    fig.savefig(path, dpi=200)
    print(f"  -> wrote {path}")


def main(args):
    out_dir = args.out or os.path.join(ensure_results_dir(), "inoculation")
    os.makedirs(out_dir, exist_ok=True)
    records = []

    for n in SIZES:
        tag = f"n{n}"
        run_dir = os.path.join(out_dir, tag)
        print(f"\n=== inoculation with {n} examples ===")

        if n == 0:
            model_path = args.base_model
        else:
            subset_csv = os.path.join(out_dir, f"round1_{tag}.csv")
            sample_round1(n, args.seed).to_csv(subset_csv, index=False)
            model_path = run_dir
            cmd = (f"python train.py"
                   f" --train {subset_csv}"
                   f" --dev {args.external_dir}/sst3_dev.csv"
                   f" --model-name {args.base_model}"
                   f" --epochs {args.epochs}"
                   f" --out {run_dir}")
            print(f"  $ {cmd}")
            if os.system(cmd) != 0:
                raise SystemExit(f"training failed at n={n}")

        eval_cmd = (f"python evaluate.py --model {model_path}"
                    f" --which model0 --external-dir {args.external_dir}")
        print(f"  $ {eval_cmd}")
        os.system(eval_cmd)

        df = pd.read_csv(os.path.join(ensure_results_dir(), "table02b.csv"),
                         index_col=0)
        for cat in ("Positive", "Negative", "Neutral"):
            for col in df.columns:
                records.append({"n": n, "category": cat, "split": col,
                                "f1": df.loc[cat, col]})
        pd.DataFrame(records).to_csv(
            os.path.join(out_dir, "inoculation_results.csv"), index=False)

    plot(pd.DataFrame(records), out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base-model", required=True)
    p.add_argument("--external-dir", default="data")
    p.add_argument("--out", default=None)
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--seed", type=int, default=1)
    main(p.parse_args())
