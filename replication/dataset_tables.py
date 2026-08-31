import pandas as pd

from common import R1, R2, SST_DEV, load_dataset, save_table

ROW_ORDER = ["Positive", "Negative", "Neutral", "Mixed", "No Majority", "Total"]

SST_TO_TERNARY = {"0": "Negative", "1": "Negative", "2": "Neutral",
                  "3": "Positive", "4": "Positive"}

PUBLISHED = {
    "table04": {"Positive": [130045, 21391, 1200, 1200],
                "Negative": [86486, 14021, 1200, 1200],
                "Neutral": [215935, 45076, 1200, 1200],
                "Mixed": [39829, 3900, 0, 0],
                "No Majority": [0, 10071, 0, 0],
                "Total": [472295, 94459, 3600, 3600]},
    "table07": {"Positive": [32551, 6038, 240, 240],
                "Negative": [24994, 4579, 240, 240],
                "Neutral": [16365, 2448, 240, 240],
                "Mixed": [18765, 3334, 0, 0],
                "No Majority": [0, 2136, 0, 0],
                "Total": [92675, 18535, 720, 720]},
    "table10": {"Positive": [367, 2, 64], "Negative": [5, 359, 57],
                "Neutral": [23, 8, 44], "Mixed": [34, 35, 39],
                "No Majority": [15, 24, 25]},
}


def majority_counts(dataset):
    labels = [d["gold_label"] if d["gold_label"] else "No Majority"
              for d in dataset]
    s = pd.Series(labels).value_counts()
    s.index = [i.title() if i != "No Majority" else i for i in s.index]
    return s


def response_counts(dataset):
    labels = []
    for d in dataset:
        for cat, workers in d["label_distribution"].items():
            labels.extend([cat.title()] * len(workers))
    return pd.Series(labels).value_counts()


def round_table(splits):
    df = pd.DataFrame({
        ("Dist", "Train"): response_counts(splits["train"]),
        ("Majority Label", "Train"): majority_counts(splits["train"]),
        ("Majority Label", "Dev"): majority_counts(splits["dev"]),
        ("Majority Label", "Test"): majority_counts(splits["test"]),
    })
    df = df.reindex([r for r in ROW_ORDER if r != "Total"])
    df = df.fillna(0).astype(int)
    df.loc["Total"] = df.sum()
    df.columns = pd.MultiIndex.from_tuples(list(df.columns))
    return df


def sst_comparison():
    data = load_dataset(SST_DEV)
    rows = [{"ours": d["gold_label"].title() if d["gold_label"]
             else "No Majority",
             "sst3": SST_TO_TERNARY[str(d["sst_label"])]} for d in data]
    tab = pd.crosstab(pd.DataFrame(rows)["ours"], pd.DataFrame(rows)["sst3"])
    tab = tab.reindex(index=ROW_ORDER[:-1],
                      columns=["Positive", "Negative", "Neutral"])
    return tab.fillna(0).astype(int)


def compare(name, tab):
    bad = 0
    for row, want in PUBLISHED[name].items():
        got = list(tab.loc[row].values)
        if got != want:
            bad += 1
            print(f"  DIFF {name} {row}: paper={want} ours={got}")
    print(f"  {name}: {bad} row(s) differ from the paper")


def main():
    r1 = {k: load_dataset(v) for k, v in R1.items()}
    r2 = {k: load_dataset(v) for k, v in R2.items()}

    print("Table 4: Round 1 dataset")
    t4 = round_table(r1)
    print(t4.to_string())
    save_table(t4, "table04_round1_splits", float_format="%d")

    print("\nTable 7: Round 2 dataset")
    t7 = round_table(r2)
    print(t7.to_string())
    save_table(t7, "table07_round2_splits", float_format="%d")

    print("\nTable 10: SST-3 dev vs DynaSent revalidation")
    t10 = sst_comparison()
    print(t10.to_string())
    save_table(t10, "table10_sst_comparison", float_format="%d")

    print("\nCheck against published values")
    compare("table04", t4)
    compare("table07", t7)
    compare("table10", t10)


if __name__ == "__main__":
    main()
