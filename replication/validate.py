import argparse
import json
import os
import random
import re
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"

LABELS = ("positive", "negative", "no sentiment", "mixed sentiment")
TERNARY = {"positive": "positive",
           "negative": "negative",
           "no sentiment": "neutral"}

DEFAULT_JUDGES = [
    "llama3.1:8b",
    "qwen2.5:7b",
    "mistral:7b",
    "gemma2:9b",
    "phi3:mini",
]

# Verbatim from validation-hit-contents.html in the DynaSent repository,
# with one adaptation: the original showed ten sentences per assignment,
# this shows one. Everything else is the authors' own wording, including
# all six worked examples.
INSTRUCTIONS = """\
You will be shown a sentence from a review of a product or service. Your
task is to choose from one of four labels:

Positive: The sentence conveys information about the author's positive
evaluative sentiment.
Negative: The sentence conveys information about the author's negative
evaluative sentiment.
No sentiment: The sentence does not convey anything about the author's
positive or negative sentiment.
Mixed sentiment: The sentence conveys a mix of positive and negative
sentiment with no clear overall sentiment.

Here are some simple examples of the labels:

Sentence: This is an under-appreciated little gem of a movie.
This is Positive because it expresses a positive overall opinion.

Sentence: I asked for my steak medium-rare, and they delivered this
perfectly!
This is Positive because it puts a positive spin on an aspect of the
author's experience.

Sentence: The screen on this device is a little too bright.
This is Negative because it negatively evaluates an aspect of the product.

Sentence: The book is 972 pages long.
This is No sentiment because it describes a factual matter with no
evaluative component.

Sentence: The waiting room is drab but the examination rooms are cheery
enough.
This is Mixed sentiment because two different sentiment evaluations are
balanced against each other.

Sentence: The entrees are delicious, but the service is so bad that it's
not worth going.
This is Negative because the negative statement outweighs the positive one.
"""

PROMPT = """{instructions}

Sentence: {sentence}

Reply with exactly one of these four labels and nothing else:
Positive
Negative
No sentiment
Mixed sentiment

Answer:"""

_CANON = {
    "positive": "positive",
    "negative": "negative",
    "no sentiment": "no sentiment",
    "nosentiment": "no sentiment",
    "no_sentiment": "no sentiment",
    "none": "no sentiment",
    "neutral": "no sentiment",
    "mixed sentiment": "mixed sentiment",
    "mixed": "mixed sentiment",
}


def ask(model, prompt, temperature, timeout, retries=3):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": 12},
    }).encode("utf8")
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"})
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf8")).get("response", "")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            if attempt == retries - 1:
                return f"__ERROR__ {e}"
            time.sleep(2 ** attempt)
    return "__ERROR__ unreachable"


def parse(raw):
    if not raw or raw.startswith("__ERROR__"):
        return None
    text = raw.strip().lower()

    for line in [l for l in text.splitlines() if l.strip()]:
        key = line.strip().strip(".,:;!?\"'*_-` ")
        if key in _CANON:
            return _CANON[key]

    found = set()
    if re.search(r"\bmixed\b", text):
        found.add("mixed sentiment")
    if re.search(r"\bno[\s_]?sentiment\b|\bneutral\b", text):
        found.add("no sentiment")
    if re.search(r"\bpositive\b", text):
        found.add("positive")
    if re.search(r"\bnegative\b", text):
        found.add("negative")
    if found == {"mixed sentiment"}:
        return "mixed sentiment"
    if found == {"no sentiment"}:
        return "no sentiment"
    if len(found) == 1:
        return found.pop()
    if found == {"positive", "negative", "mixed sentiment"}:
        return "mixed sentiment"
    return None


def load_items(path, limit, seed):
    items = []
    if path.lower().endswith(".jsonl"):
        with open(path, encoding="utf8") as f:
            for i, line in enumerate(f):
                if not line.strip():
                    continue
                r = json.loads(line)
                s = r.get("sentence") or r.get("text")
                if not s:
                    continue
                items.append({
                    "id": r.get("text_id") or r.get("sentence_id") or f"row{i}",
                    "sentence": s,
                    "gold": r.get("gold_label"),
                    "star_label": None,
                    "review_rating": None,
                })
    else:
        import csv
        with open(path, encoding="utf8", newline="") as f:
            for i, r in enumerate(csv.DictReader(f)):
                s = (r.get("sentence") or "").strip()
                if not s:
                    continue
                items.append({
                    "id": r.get("id") or f"row{i}",
                    "sentence": s,
                    "gold": None,
                    "star_label": r.get("star_label"),
                    "review_rating": r.get("review_rating"),
                })
    items = [it for it in items
             if it["gold"] in (None, "positive", "negative", "neutral")]
    random.Random(seed).shuffle(items)
    return items[:limit] if limit else items


def load_done(path):
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                r = json.loads(line)
                done.add((r["id"], r["judge"]))
            except Exception:
                continue
    return done


def main(a):
    items = load_items(a.input, a.limit, a.seed)
    judges = a.judges.split(",") if a.judges else DEFAULT_JUDGES
    os.makedirs(os.path.dirname(a.ballots) or ".", exist_ok=True)
    done = load_done(a.ballots)

    total = len(items) * len(judges)
    print(f"{len(items):,} sentences x {len(judges)} judges = {total:,} judgements")
    print(f"{len(done):,} already recorded, {max(total - len(done), 0):,} to do")
    if a.dry_run:
        print("\nDRY RUN - the exact prompt that will be sent:\n")
        print("=" * 70)
        print(PROMPT.format(instructions=INSTRUCTIONS,
                            sentence=items[0]["sentence"]))
        print("=" * 70)
        return

    t0 = time.time()
    n = 0
    with open(a.ballots, "a", encoding="utf8") as out:
        for judge in judges:
            for it in items:
                if (it["id"], judge) in done:
                    continue
                raw = ask(judge,
                          PROMPT.format(instructions=INSTRUCTIONS,
                                        sentence=it["sentence"]),
                          a.temperature, a.timeout)
                label = parse(raw)
                out.write(json.dumps({
                    "id": it["id"],
                    "judge": judge,
                    "label": label,
                    "ternary": TERNARY.get(label),
                    "raw": raw[:200],
                    "sentence": it["sentence"],
                    "gold": it["gold"],
                    "star_label": it["star_label"],
                    "review_rating": it["review_rating"],
                }, ensure_ascii=False) + "\n")
                out.flush()
                n += 1
                if n % 25 == 0:
                    rate = n / (time.time() - t0)
                    left = (total - len(done) - n) / rate if rate else 0
                    print(f"  {n:,} done  {rate:.1f}/s  ~{left/60:.0f} min left")

    print(f"\nwrote {a.ballots}")
    print(f"{n:,} new judgements in {(time.time()-t0)/60:.1f} minutes")
    print(f"\nNext: python replication\\agreement.py --ballots {a.ballots}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--ballots", required=True)
    p.add_argument("--judges", default="")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--timeout", type=int, default=120)
    p.add_argument("--seed", type=int, default=8240)
    p.add_argument("--dry-run", action="store_true")
    main(p.parse_args())
