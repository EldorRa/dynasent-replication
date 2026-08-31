import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch.utils.data import Dataset
from transformers import (AutoConfig, AutoModelForSequenceClassification,
                          AutoTokenizer, Trainer, TrainingArguments)

LABELS = ["negative", "positive", "neutral"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}


class SentimentDataset(Dataset):
    def __init__(self, encodings, labels, soft=False):
        self.encodings = encodings
        self.labels = labels
        self.soft = soft

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(
            self.labels[idx],
            dtype=torch.float if self.soft else torch.long)
        return item


class SoftLabelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logp = torch.nn.functional.log_softmax(outputs.logits, dim=-1)
        loss = -(labels * logp).sum(dim=-1).mean()
        return (loss, outputs) if return_outputs else loss


def read_csv(paths, soft=False):
    df = pd.concat([pd.read_csv(p) for p in paths], ignore_index=True)
    df = df.dropna(subset=["sentence"])
    X = df["sentence"].astype(str).tolist()
    if soft:
        y = df[LABELS].values.astype("float32")
        y = y / y.sum(axis=1, keepdims=True)
    else:
        y = [LABEL2ID[l] for l in df["label"]]
    return X, y


def metrics_fn(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    if labels.ndim > 1:
        labels = np.argmax(labels, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "f1_positive": f1_score(labels, preds, average=None,
                                labels=[LABEL2ID["positive"]])[0],
        "f1_negative": f1_score(labels, preds, average=None,
                                labels=[LABEL2ID["negative"]])[0],
        "f1_neutral": f1_score(labels, preds, average=None,
                               labels=[LABEL2ID["neutral"]])[0],
    }


def main(args):
    tok = AutoTokenizer.from_pretrained(args.model_name)

    X_train, y_train = read_csv(args.train, soft=args.soft)
    X_dev, y_dev = read_csv(args.dev, soft=False)
    print(f"train {len(X_train):,}   dev {len(X_dev):,}")

    enc_train = tok(X_train, truncation=True, padding="max_length",
                    max_length=args.max_length)
    enc_dev = tok(X_dev, truncation=True, padding="max_length",
                  max_length=args.max_length)

    config = AutoConfig.from_pretrained(
        args.model_name, num_labels=3,
        hidden_dropout_prob=0.1, attention_probs_dropout_prob=0.1,
        id2label={i: l for l, i in LABEL2ID.items()}, label2id=LABEL2ID)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, config=config)

    targs = TrainingArguments(
        output_dir=args.out,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 4,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        fp16=torch.cuda.is_available(),
        logging_steps=200,
        seed=args.seed,
        report_to=[],
    )

    cls = SoftLabelTrainer if args.soft else Trainer
    trainer = cls(
        model=model, args=targs,
        train_dataset=SentimentDataset(enc_train, y_train, soft=args.soft),
        eval_dataset=SentimentDataset(enc_dev, y_dev, soft=False),
        compute_metrics=metrics_fn)

    trainer.train()
    trainer.save_model(args.out)
    tok.save_pretrained(args.out)
    print(trainer.evaluate())
    torch.save(model.state_dict(), os.path.join(args.out, "pytorch_model.bin"))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", nargs="+", required=True)
    p.add_argument("--dev", nargs="+", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--model-name", default="roberta-base")
    p.add_argument("--soft", action="store_true")
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    main(p.parse_args())
