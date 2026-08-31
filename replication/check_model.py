import argparse
import time

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from common import R1, TERNARY_CATS, load_dataset
from evaluate import load_model

CLASSES = ["negative", "positive", "neutral"]

DEMO = [
    ("superb", "positive"),
    ("They said the experience would be amazing, and they were right!",
     "positive"),
    ("They said the experience would be amazing, and they were wrong!",
     "negative"),
]


def main(args):
    print("1. Loading the model")
    t0 = time.time()
    model = load_model(args.model)
    tok = AutoTokenizer.from_pretrained("roberta-base")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()
    print(f"   loaded in {time.time() - t0:.1f}s on {device}")
    if device.type == "cuda":
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("   No GPU detected; this will be slow.")

    def predict(sentences, batch_size=32):
        out = []
        for i in range(0, len(sentences), batch_size):
            enc = tok(sentences[i:i + batch_size], max_length=128,
                      padding="max_length", truncation=True,
                      return_tensors="pt").to(device)
            with torch.no_grad():
                logits = model(**enc).logits
            probs = F.softmax(logits, dim=1).cpu()
            out.extend(CLASSES[j] for j in probs.argmax(1).tolist())
        return out

    print("\n2. Demo sentences")
    preds = predict([s for s, _ in DEMO])
    all_ok = True
    for (sentence, expected), got in zip(DEMO, preds):
        ok = got == expected
        all_ok &= ok
        print(f"   [{'ok ' if ok else 'BAD'}] {got:<9} "
              f"(expected {expected:<9}) {sentence[:52]}")
    if not all_ok:
        raise SystemExit("   Demo predictions are wrong: check the label "
                         "order and that the classifier head loaded.")
    print("   All three correct.")

    print("\n3. Speed check on 500 sentences")
    data = load_dataset(R1["dev"], labels=TERNARY_CATS)[:500]
    t0 = time.time()
    predict([d["sentence"] for d in data])
    dt = time.time() - t0
    rate = 500 / dt
    print(f"   {dt:.1f}s for 500 = {rate:.0f} sentences/second")
    print(f"   Round 1 dev+test (7,200):  ~{7200 / rate / 60:.1f} min")
    print(f"   Yelp dev+test (50,000):    ~{50000 / rate / 60:.1f} min")
    print(f"   Amazon dev+test (650,000): ~{650000 / rate / 3600:.1f} hours")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    main(p.parse_args())
