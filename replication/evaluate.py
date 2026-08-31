import argparse
import os
import sys

import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer)

from common import R1, R2, TERNARY_CATS, load_dataset, save_table

CLASSES = ["negative", "positive", "neutral"]

ORDER = ["SST-3 Dev", "SST-3 Test", "Yelp Dev", "Yelp Test",
         "Amazon Dev", "Amazon Test", "Round 1 Dev", "Round 1 Test",
         "Round 2 Dev", "Round 2 Test"]

PAPER = {
    "model0": {
        "SST-3 Dev": [85.1, 84.1, 45.4, 71.5],
        "SST-3 Test": [89.0, 84.1, 43.5, 72.2],
        "Yelp Dev": [88.3, 88.8, 58.2, 78.4],
        "Yelp Test": [90.5, 89.1, 59.4, 79.7],
        "Amazon Dev": [89.1, 86.6, 53.9, 76.5],
        "Amazon Test": [89.4, 86.6, 53.7, 76.6],
        "Round 1 Dev": [33.3, 33.3, 33.3, 33.3],
        "Round 1 Test": [33.3, 33.3, 33.3, 33.3],
        "Round 2 Dev": [58.4, 61.0, 38.4, 52.6],
        "Round 2 Test": [63.0, 63.1, 44.3, 56.8],
    },
    "model1": {
        "SST-3 Dev": [84.6, 82.7, 40.0, 69.1],
        "SST-3 Test": [88.6, 84.4, 45.2, 72.7],
        "Yelp Dev": [80.0, 79.5, 56.7, 72.1],
        "Yelp Test": [83.1, 79.6, 56.6, 73.1],
        "Amazon Dev": [83.3, 78.7, 55.5, 72.5],
        "Amazon Test": [83.3, 78.8, 55.4, 72.5],
        "Round 1 Dev": [81.0, 80.5, 83.1, 81.5],
        "Round 1 Test": [80.4, 80.2, 83.5, 81.4],
        "Round 2 Dev": [33.3, 33.3, 33.3, 33.3],
        "Round 2 Test": [33.3, 33.3, 33.3, 33.3],
    },
}


def load_model(model_path):
    config = AutoConfig.from_pretrained(
        "roberta-base", num_labels=3, finetuning_task="sst3")

    if os.path.isdir(model_path):
        return AutoModelForSequenceClassification.from_pretrained(
            model_path, config=config)

    model = AutoModelForSequenceClassification.from_config(config)
    try:
        state = torch.load(model_path, map_location="cpu", weights_only=False)
    except TypeError:
        state = torch.load(model_path, map_location="cpu")

    if not isinstance(state, dict):
        return state

    state = {k[7:] if k.startswith("module.") else k: v
             for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"  WARNING missing keys: {list(missing)[:6]}")
    if unexpected:
        print(f"  ignored {len(unexpected)} unexpected key(s)")
    if any("classifier" in k for k in missing):
        raise SystemExit("  The classifier head did not load.")
    return model


class Predictor:
    def __init__(self, model_path, batch_size=64, max_length=128, device=None):
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.tokenizer = AutoTokenizer.from_pretrained("roberta-base")
        self.model = load_model(model_path).to(self.device).eval()
        print(f"  loaded {model_path} on {self.device}")

    def predict(self, strings):
        out = []
        for i in range(0, len(strings), self.batch_size):
            enc = self.tokenizer(strings[i: i + self.batch_size],
                                 max_length=self.max_length,
                                 padding="max_length", truncation=True,
                                 return_tensors="pt").to(self.device)
            with torch.no_grad():
                logits = self.model(**enc).logits
            probs = F.softmax(logits, dim=1).cpu()
            out.extend(CLASSES[j] for j in probs.argmax(1).tolist())
            if (i // self.batch_size) % 20 == 0:
                print(f"    {min(i + self.batch_size, len(strings)):,}"
                      f"/{len(strings):,}", file=sys.stderr)
        return out


def f1_column(y_true, y_pred):
    scores = f1_score(y_true, y_pred, labels=list(TERNARY_CATS), average=None)
    col = {c.title(): s * 100 for c, s in zip(TERNARY_CATS, scores)}
    col["Macro avg"] = sum(col.values()) / 3
    return col


def dynasent_splits():
    out = {}
    for name, paths in (("Round 1", R1), ("Round 2", R2)):
        for split in ("dev", "test"):
            data = load_dataset(paths[split], labels=TERNARY_CATS)
            out[f"{name} {split.title()}"] = (
                [d["sentence"] for d in data],
                [d["gold_label"] for d in data])
    return out


def external_splits(data_dir):
    out = {}
    for name, stem in (("SST-3", "sst3"), ("Yelp", "yelp"),
                       ("Amazon", "amazon")):
        for split in ("dev", "test"):
            path = os.path.join(data_dir, f"{stem}_{split}.csv")
            if not os.path.exists(path):
                continue
            df = pd.read_csv(path).dropna(subset=["sentence"])
            out[f"{name} {split.title()}"] = (
                df["sentence"].astype(str).tolist(), df["label"].tolist())
    return out


def main(args):
    predictor = Predictor(args.model, batch_size=args.batch_size)

    splits = dynasent_splits()
    if args.external_dir:
        splits.update(external_splits(args.external_dir))

    columns = {}
    for name in ORDER:
        if name not in splits:
            continue
        X, y = splits[name]
        print(f"  predicting {name} ({len(X):,})")
        columns[name] = f1_column(y, predictor.predict(X))

    df = pd.DataFrame(columns).reindex(
        ["Positive", "Negative", "Neutral", "Macro avg"])
    print("\n" + df.round(1).to_string())

    table = "table02b" if args.which == "model0" else "table05b"
    save_table(df.round(1), table)

    print(f"\nvs published {table}")
    paper = PAPER[args.which]
    for name in df.columns:
        want = paper[name]
        got = [round(g, 1) for g in df[name].tolist()]
        deltas = [f"{g - w:+.1f}" for g, w in zip(got, want)]
        print(f"  {name:<14} paper={want}  ours={got}  delta={deltas}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--which", choices=["model0", "model1"], required=True)
    p.add_argument("--external-dir", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    main(p.parse_args())
