import random

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support
from statsmodels.stats.inter_rater import fleiss_kappa

from common import R1, R2, ALL_CATS, TERNARY_CATS, load_dataset, save_table

RANDOM_STATE = 1

PUBLISHED = {
    "table06": {"f1": {"Positive": (88.1, 87.8), "Negative": (89.2, 89.3),
                       "Neutral": (86.6, 86.9), "Macro avg": (88.0, 88.0)},
                "kappa": (0.616, 0.615)},
    "table09": {"f1": {"Positive": (91.0, 90.9), "Negative": (91.2, 91.0),
                       "Neutral": (88.9, 88.2), "Macro avg": (90.4, 90.0)},
                "kappa": (0.681, 0.667)},
}


def estimate_human_f1(split, cats=TERNARY_CATS, random_state=RANDOM_STATE):
    random.seed(random_state)
    y, all_responses = [], []
    for d in split:
        if d["gold_label"] is None:
            continue
        y.append(d["gold_label"])
        responses = [c for c, vals in d["label_distribution"].items()
                     for _ in range(len(vals))]
        random.shuffle(responses)
        all_responses.append(responses)

    all_scores = []
    for i in range(len(all_responses[0])):
        ann = [r[i] for r in all_responses]
        all_scores.append(
            np.array(precision_recall_fscore_support(y, ann, labels=cats)))

    mu = np.mean(all_scores, axis=0)
    df = pd.DataFrame(mu.T, index=cats,
                      columns=["precision", "recall", "F1", "support"])
    df.loc["Macro avg"] = df.mean(axis=0)
    return df


def kappa(split, cats=ALL_CATS):
    mat = np.zeros((len(split), len(cats)))
    for i, d in enumerate(split):
        for j, cat in enumerate(cats):
            mat[i][j] = len(d["label_distribution"][cat])
    return fleiss_kappa(mat)


def build(splits, table, title):
    dev = load_dataset(splits["dev"])
    test = load_dataset(splits["test"])
    f1 = pd.DataFrame({
        "Dev": estimate_human_f1(dev)["F1"] * 100,
        "Test": estimate_human_f1(test)["F1"] * 100,
    })
    f1.index = [i.title() if i != "Macro avg" else i for i in f1.index]

    paper = PUBLISHED[table]
    print(f"\n{table}: {title}")
    print(f1.round(1).to_string())
    print(f"  Fleiss kappa  dev={kappa(dev):.3f}  test={kappa(test):.3f}"
          f"   (paper: {paper['kappa'][0]} / {paper['kappa'][1]})")
    save_table(f1.round(1), table)

    print("  vs paper:")
    for row, (want_dev, want_test) in paper["f1"].items():
        got = f1.loc[row]
        print(f"    {row:<10} dev paper={want_dev} ours={got['Dev']:.1f}"
              f" | test paper={want_test} ours={got['Test']:.1f}")
    return f1


if __name__ == "__main__":
    build(R1, "table06", "Round 1 human performance")
    build(R2, "table09", "Round 2 human performance")
