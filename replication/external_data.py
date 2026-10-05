import argparse
import io
import os
import re
import time
import urllib.request
import zipfile

import pandas as pd

from common import R1, SST_DEV, load_dataset, save_table

STAR_TO_TERNARY = {1: "negative", 2: "negative", 3: "neutral",
                   4: "positive", 5: "positive"}
SST_TO_TERNARY = {0: "negative", 1: "negative", 2: "neutral",
                  3: "positive", 4: "positive"}

SST_TREES_URL = "https://nlp.stanford.edu/sentiment/trainDevTestTrees_PTB.zip"

TABLE1_PAPER = {
    ("SST-3", "Dev"): {"Positive": 444, "Negative": 428, "Neutral": 228,
                       "Total": 1100},
    ("SST-3", "Test"): {"Positive": 909, "Negative": 912, "Neutral": 389,
                        "Total": 2210},
    ("Yelp", "Dev"): {"Positive": 9577, "Negative": 10222, "Neutral": 5201,
                      "Total": 25000},
    ("Yelp", "Test"): {"Positive": 10423, "Negative": 9778, "Neutral": 4799,
                       "Total": 25000},
    ("Amazon", "Dev"): {"Positive": 130631, "Negative": 129108,
                        "Neutral": 65261, "Total": 325000},
    ("Amazon", "Test"): {"Positive": 129369, "Negative": 130892,
                         "Neutral": 64739, "Total": 325000},
}

TRAINING_PAPER = {"cr_train": ("CR", 3771), "imdb_train": ("IMDB", 25000)}


def round1_distributional(dataset):
    counts = {"positive": 0, "negative": 0, "neutral": 0}
    rows = []
    for d in dataset:
        dist = d["label_distribution"]
        for cat in ("positive", "negative", "neutral"):
            n = len(dist[cat])
            counts[cat] += n
            rows.extend([(d["sentence"], cat)] * n)
        n_mixed = len(dist["mixed"])
        counts["positive"] += n_mixed
        counts["negative"] += n_mixed
        rows.extend([(d["sentence"], "positive")] * n_mixed)
        rows.extend([(d["sentence"], "negative")] * n_mixed)
    return counts, rows


def check_round1_column():
    counts, rows = round1_distributional(load_dataset(R1["train"]))
    doubled = {k: v * 2 for k, v in counts.items()}
    paper = {"positive": 339748, "negative": 252630, "neutral": 431870}
    total = sum(doubled.values())

    print("Table 5a, Round 1 column")
    for cat in ("positive", "negative", "neutral"):
        flag = "ok " if doubled[cat] == paper[cat] else "DIFF"
        print(f"  {flag} {cat:<9} paper={paper[cat]:>9,}  ours={doubled[cat]:>9,}")
    flag = "ok " if total == 1024248 else "DIFF"
    print(f"  {flag} {'Total':<9} paper={1024248:>9,}  ours={total:>9,}")
    print(f"  one copy = {len(rows):,} rows")
    return rows


def check_sst_dev_column():
    data = load_dataset(SST_DEV)
    raw = pd.Series([SST_TO_TERNARY[int(d["sst_label"])] for d in data])
    seen, dedup = set(), []
    for d in data:
        if d["sentence"] in seen:
            continue
        seen.add(d["sentence"])
        dedup.append(SST_TO_TERNARY[int(d["sst_label"])])

    df = pd.DataFrame({
        "all rows": raw.value_counts(),
        "deduplicated": pd.Series(dedup).value_counts(),
        "paper (Table 1)": pd.Series({"positive": 444, "negative": 428,
                                      "neutral": 228})})
    df.loc["Total"] = df.sum()
    print("\nTable 1, SST-3 dev column")
    print(df.to_string())
    return df


def _dump(rows, out_dir, name):
    path = os.path.join(out_dir, f"{name}.csv")
    pd.DataFrame(rows, columns=["sentence", "label"]).to_csv(path, index=False)
    print(f"    wrote {path}  ({len(rows):,} rows)")
    return path


def _halve_by_line(records, to_row):
    n = len(records)
    half = n // 2
    dev = [to_row(records[i]) for i in range(half)]
    test = [to_row(records[i]) for i in range(half, n)]
    return dev, test


def _hf(name, retries=8, **kw):
    from datasets import load_dataset as hf_load
    delay = 10
    for attempt in range(1, retries + 1):
        try:
            return hf_load(name, **kw)
        except RuntimeError as e:
            if "no longer supported" in str(e):
                raise SystemExit(
                    f"  '{name}' is script-based and datasets v4 removed "
                    f"script support. Use a parquet mirror or "
                    f"`pip install \"datasets<4\"`.")
            raise
        except Exception as e:
            if attempt == retries:
                raise SystemExit(
                    f"  Gave up after {retries} attempts ({type(e).__name__}). "
                    f"Partial downloads are cached; run the command again.")
            print(f"  attempt {attempt}/{retries} failed "
                  f"({type(e).__name__}); retrying in {delay}s")
            time.sleep(delay)
            delay = min(delay * 2, 120)


def _parse_ptb(line):
    tokens = line.replace("(", " ( ").replace(")", " ) ").split()
    pos = 0

    def node():
        nonlocal pos
        pos += 1
        label = int(tokens[pos])
        pos += 1
        children, word = [], None
        while tokens[pos] != ")":
            if tokens[pos] == "(":
                children.append(node())
            else:
                word = tokens[pos]
                pos += 1
        pos += 1
        return (label, word, children)

    return node()


def _subtrees(line):
    out = []

    def walk(n):
        label, word, children = n
        if word is not None:
            out.append((word, label))
            return word
        text = " ".join(walk(c) for c in children)
        out.append((text, label))
        return text

    walk(_parse_ptb(line))
    return out


def _leaves(line):
    def walk(n):
        _, w, ch = n
        return [w] if w is not None else [t for c in ch for t in walk(c)]

    return " ".join(walk(_parse_ptb(line)))


def _detokenise(s):
    s = s.replace("\\/", "/")
    s = re.sub(r" ?'' ?", '"', s)
    s = re.sub(r" ?`` ?", "``", s)
    s = s.replace(" -- ", "--")
    s = re.sub(r" (?=[.,;:!?%)\]}])", "", s)
    s = re.sub(r"(?<=[(\[{]) ", "", s)
    s = re.sub(r" (n't|'s|'m|'re|'ve|'ll|'d)\b", r"\1", s)
    return s


def build_sst(out_dir):
    print(f"SST -- downloading {SST_TREES_URL}")
    local = "trainDevTestTrees_PTB.zip"
    if os.path.exists(local):
        blob = open(local, "rb").read()
    else:
        try:
            with urllib.request.urlopen(SST_TREES_URL, timeout=120) as r:
                blob = r.read()
        except Exception as e:
            raise SystemExit(
                f"  Could not download SST: {e}\n"
                f"  Download {SST_TREES_URL} manually, put the .zip beside "
                f"this script's working directory, and run again.")
    print(f"  {len(blob) / 1e6:.1f} MB")

    zf = zipfile.ZipFile(io.BytesIO(blob))
    trees = {}
    for member in zf.namelist():
        for split in ("train", "dev", "test"):
            if member.endswith(f"{split}.txt"):
                trees[split] = zf.read(member).decode("utf-8").splitlines()
    for split in ("train", "dev", "test"):
        if split not in trees:
            raise SystemExit(f"  {split}.txt not found in the zip")
        print(f"  trees/{split}.txt: {len(trees[split]):,} sentences")

    release = load_dataset(SST_DEV)
    exact = sum(1 for d in release
                if _detokenise(_leaves(d["tree"])) == d["sentence"])
    print(f"  detokeniser agrees with the DynaSent text on "
          f"{exact:,}/{len(release):,} dev sentences "
          f"({100 * exact / len(release):.1f}%)")

    seen, dev = set(), []
    for d in release:
        if d["sentence"] in seen:
            continue
        seen.add(d["sentence"])
        dev.append((d["sentence"], SST_TO_TERNARY[int(d["sst_label"])]))
    print(f"  ok  sst3_dev: {len(dev):,} sentences taken verbatim from "
          f"sst-dev-validated.jsonl (deduplicated)")
    _dump(dev, out_dir, "sst3_dev")

    test = [(_detokenise(_leaves(line)), SST_TO_TERNARY[int(line.strip()[1])])
            for line in trees["test"] if line.strip()]
    flag = "ok " if len(test) == 2210 else "note"
    print(f"  {flag} sst3_test: {len(test):,} sentences detokenised from the "
          f"trees (SST has 2,210)")
    _dump(test, out_dir, "sst3_test")

    seen, rows = set(), []
    for line in trees["train"]:
        if not line.strip():
            continue
        for text, label in _subtrees(line):
            text = _detokenise(text)
            if text in seen:
                continue
            seen.add(text)
            rows.append((text, SST_TO_TERNARY[label]))
    _dump(rows, out_dir, "sst3_train")
    print(f"    {len(rows):,} unique phrases (paper reports 159,274)")


def build_small(out_dir):
    print("IMDB (stanfordnlp/imdb)")
    imdb = _hf("stanfordnlp/imdb", split="train")
    _dump([(x["text"], "positive" if x["label"] == 1 else "negative")
           for x in imdb], out_dir, "imdb_train")

    print("CR (SetFit/CR)")
    cr = _hf("SetFit/CR")
    _dump([(x["text"], "positive" if x["label"] == 1 else "negative")
           for split in ("train", "test") for x in cr[split]],
          out_dir, "cr_train")


def build_yelp(out_dir):
    print("Yelp (Yelp/yelp_review_full)")
    tr = _hf("Yelp/yelp_review_full", split="train")
    _dump([(x["text"], STAR_TO_TERNARY[x["label"] + 1]) for x in tr],
          out_dir, "yelp_train")
    te = _hf("Yelp/yelp_review_full", split="test")
    dev, test = _halve_by_line(
        te, lambda r: (r["text"], STAR_TO_TERNARY[r["label"] + 1]))
    _dump(dev, out_dir, "yelp_dev")
    _dump(test, out_dir, "yelp_test")


def build_amazon(out_dir):
    print("Amazon (vgaraujov/amazon_review_full)")

    def amz_row(r):
        title = r.get("title") or ""
        return (f"{title}. {r['content']}".strip(". "),
                STAR_TO_TERNARY[r["label"] + 1])

    tr = _hf("vgaraujov/amazon_review_full", split="train")
    _dump([amz_row(x) for x in tr], out_dir, "amazon_train")
    te = _hf("vgaraujov/amazon_review_full", split="test")
    dev, test = _halve_by_line(te, amz_row)
    _dump(dev, out_dir, "amazon_dev")
    _dump(test, out_dir, "amazon_test")


def training_corpus_report(out_dir):
    print("\nTraining corpora (Tables 2a / 5a)")
    for stem, (name, want) in TRAINING_PAPER.items():
        path = os.path.join(out_dir, f"{stem}.csv")
        if not os.path.exists(path):
            print(f"  {name:<6} not built")
            continue
        df = pd.read_csv(path)
        counts = df["label"].value_counts()
        flag = "ok " if len(df) == want else "DIFF"
        print(f"  {flag} {name:<6} paper={want:>6,}  ours={len(df):>6,}  "
              f"({', '.join(f'{k} {v:,}' for k, v in counts.items())})")


def table1(out_dir):
    cols, missing = {}, []
    for corpus, stem in (("SST-3", "sst3"), ("Yelp", "yelp"),
                         ("Amazon", "amazon")):
        for split in ("dev", "test"):
            path = os.path.join(out_dir, f"{stem}_{split}.csv")
            if not os.path.exists(path):
                missing.append(f"{stem}_{split}")
                continue
            s = pd.read_csv(path)["label"].value_counts()
            col = {c.title(): int(s.get(c, 0))
                   for c in ("positive", "negative", "neutral")}
            col["Total"] = sum(col.values())
            cols[(corpus, split.title())] = col

    if not cols:
        print("\nTable 1: no evaluation corpora built yet "
              "(CR and IMDB are training-only).")
        return None
    if missing:
        print(f"\n  not built yet: {', '.join(missing)}")

    df = pd.DataFrame(cols).reindex(
        ["Positive", "Negative", "Neutral", "Total"])
    df.columns = pd.MultiIndex.from_tuples(list(df.columns))
    print("\nTable 1: external assessment datasets")
    print(df.to_string())
    save_table(df, "table01_external_splits", float_format="%d")

    print("\nvs published Table 1")
    bad = checked = 0
    for key, want in TABLE1_PAPER.items():
        if key not in cols:
            continue
        for row, w in want.items():
            checked += 1
            got = cols[key][row]
            if got != w:
                bad += 1
                print(f"  DIFF {key[0]} {key[1]} {row}: "
                      f"paper={w:,} ours={got:,} ({got - w:+,})")
    print(f"  {bad} of {checked} cell(s) differ")
    return df


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--table1", action="store_true")
    p.add_argument("--only",
                   choices=["small", "sst", "yelp", "amazon", "reviews"])
    p.add_argument("--out", default="data")
    args = p.parse_args()

    if args.build:
        os.makedirs(args.out, exist_ok=True)
        if args.only in (None, "sst"):
            build_sst(args.out)
        if args.only in (None, "small"):
            build_small(args.out)
        if args.only in (None, "reviews", "yelp"):
            build_yelp(args.out)
        if args.only in (None, "reviews", "amazon"):
            build_amazon(args.out)
        training_corpus_report(args.out)
        table1(args.out)
    elif args.table1:
        training_corpus_report(args.out)
        table1(args.out)
    else:
        check_round1_column()
        check_sst_dev_column()
