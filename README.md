# DynaSent: A Replication, and an Application to New Data

COMP8240 Applications of Data Science — Macquarie University
Eldor Rakhmonaliev (48756997)

A replication of **DynaSent: A Dynamic Benchmark for Sentiment Analysis**
(Potts, Wu, Geiger & Kiela, ACL-IJCNLP 2021,
[arXiv:2012.15349](https://arxiv.org/abs/2012.15349)), followed by an
application of the paper's method to two corpora the authors never saw.

---

## Relationship to the original repository

This repository **descends from the authors' own release** at
[cgpotts/dynasent](https://github.com/cgpotts/dynasent), so its early
history and the `dynasent-v1.1/` data are theirs, not mine.

Everything under `replication/` is my own work, as are everything in
`results/` and `replication/results/`, the notebooks, and this README.

---

## Headline findings

**All ten tables attempted reproduce exactly** — but two of them only
after corrections the paper does not specify.

| | What went wrong | Effect |
|---|---|---|
| **SST-3** | The Treebank ships tokenised. A whitespace join gives `the food was good .`; the model was trained on `the food was good.` | Neutral F1 **12.6 → 45.7** against a published 45.4 |
| **Amazon** | The corpus stores review title and body in separate columns. The paper does not say which it used. | Body alone reproduces all 16 cells exactly; title+body runs **+0.7 to +2.9** high |

On new data, the paper's central claim holds. The *neutral gap* — the
margin by which the mean of positive and negative F1 exceeds neutral F1 —
falls from **19.6 to 1.4** on SemEval-2017 tweets and from **38.3 to 16.1**
on a Google Local corpus I built and labelled with a five-model judge
panel calibrated at 83.5% agreement against DynaSent's own human
annotations.

---

## Setup

```bash
git clone https://github.com/EldorRa/dynasent-replication.git
cd dynasent-replication
python -m venv venv
venv\Scripts\activate          # Windows;  source venv/bin/activate on Unix
pip install -r requirements.txt
```

A CUDA GPU is strongly recommended. Scoring all external corpora is about
1.4 million forward passes and takes roughly three hours per model on an
RTX 4070 Laptop; on CPU it is impractical.

### Data and models not in this repository

Three things are excluded for size and must be fetched separately.

| What | Where it goes | Why it is not here |
|---|---|---|
| `dynasent_model0.bin`, `dynasent_model1.bin` | `models/` | Linked from the [official DynaSent repository](https://github.com/cgpotts/dynasent) |
| External corpora (Yelp, Amazon, IMDB, SST, CR) | `data/` | Rebuilt by `external_data.py --build`; `amazon_train.csv` alone is 1.28 GB |
| `review-Hawaii.json.gz` | repository root | 290 MB. The Hawaii slice of the [Google Local review data](https://jiachengli1995.github.io/google/index.html) (McAuley lab, UCSD) |

`dynasent-v1.1/` **is** included, inherited from the authors' release.

The Google Local file is downloaded from
[mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal](https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/).
Its authors ask that two papers be cited — Li, Shang & McAuley (ACL 2022)
and Yan et al. (SIGIR 2023); both are in `references.bib`.

Each review record carries `user_id`, `name`, `time`, `rating`, `text`,
`pics`, `resp` and `gmap_id`. **`name` is the reviewer's own name.**

`google_local.py` reads `text`, `rating`, `gmap_id` and `time`. The
reviewer's name and `user_id` are never read. The intermediate
`data/google_local_prompts.csv` keeps `gmap_id` and `review_time` as
business-level keys, and that file is not committed. The published
`results/validated_*.csv` carry only the sentence, the panel labels, the
star-derived label and the star rating — no business key, no timestamp.

### For the judge panel only

The annotation stage runs five local language models through
[Ollama](https://ollama.com), entirely offline and at no cost:

```bash
ollama pull llama3.1:8b
ollama pull qwen2.5:7b
ollama pull mistral:7b
ollama pull gemma2:9b
ollama pull phi3:mini
```

---

## Part 1 — Reproducing the paper

Run in this order. Scripts not listed with flags take no arguments.

```bash
# Sanity checks
python replication/check_setup.py
python replication/check_model.py --model models/dynasent_model1.bin

# Build the five external training corpora into data/   (slow: ~1 GB written)
python replication/external_data.py --build

# Table 1 — external corpus statistics, 24 cells
python replication/external_data.py --table1

# Tables 4, 7, 10 — dataset composition, both rounds
python replication/dataset_tables.py

# Tables 3, 8 — worked examples from each round
python replication/example_tables.py

# Tables 6, 9 — human performance and Fleiss' kappa
python replication/human_performance.py

# Tables 2b and 5b — model scores on every corpus
# --external-dir is REQUIRED; without it the external corpora are skipped
python replication/evaluate.py --model models/dynasent_model0.bin \
    --which model0 --external-dir data
python replication/evaluate.py --model models/dynasent_model1.bin \
    --which model1 --external-dir data
```

### The Amazon text-field diagnostic

`external_data.py` builds the Amazon splits from the `content` column
alone. To reproduce the comparison in §3.5 of the report, keep a
title+body build alongside it as `data/amazon_test.titlecontent.csv`,
then:

```bash
python replication/amazon_ab.py --sample 25000
```

This scores both builds on identical row indices, so the text field is the
only thing that differs. Output: `replication/results/table05c_amazon_ab.csv`.

### Optional

```bash
# Retrain from scratch instead of using the released binaries
python replication/train.py --train <files> --dev <files> --out models/mine.bin

# Inoculation-by-fine-tuning experiment (not part of the replicated tables)
python replication/inoculation.py --base-model models/dynasent_model0.bin
```

---

## Part 2 — New data

### New existing dataset: SemEval-2017 Task 4A

```bash
python replication/new_data_eval.py --dataset tweet_eval
```

### Constructed dataset: Google Local reviews (Hawaii)

```bash
# 1. Sample balanced candidate sentences from the raw reviews
python replication/google_local.py --input review-Hawaii.json.gz \
    --per-star 400 --out data/google_local_prompts.csv

# 2. Keep those where Model 1 contradicts the star-derived label
python replication/harvest_adversarial.py \
    --prompts data/google_local_prompts.csv \
    --out data/round3_candidates.csv

# 3. Label them with the five-model judge panel (Ollama must be running).
#    Resumable: re-running skips judgements already in the ballots file.
python replication/validate.py --input data/round3_candidates.csv \
    --ballots results/ballots_round3.jsonl

# 4. Fleiss' kappa, majority vote, and the validated gold labels
python replication/agreement.py --ballots results/ballots_round3.jsonl \
    --out results/validated_round3.csv

# 5. Score both models against the validated gold
python replication/score_validated.py \
    --validated results/validated_round3.csv \
    --out results/table06_new_data.csv
```

**Check the panel before trusting it.** Run steps 3 and 4 first on
DynaSent's own development set, where human labels already exist:

```bash
python replication/validate.py --input dynasent-v1.1/dynasent-v1.1-round01-yelp-dev.jsonl \
    --limit 200 --ballots results/ballots_calibration.jsonl
python replication/agreement.py --ballots results/ballots_calibration.jsonl
```

This reports the panel's agreement with human gold. Mine was 83.5%
(macro F1 83.4), against the 88.0 the authors estimate for a human
annotator. That number is what makes the substitution defensible; without
it the panel is an unvalidated instrument.

### Avoiding a selection effect

The harvested candidates were chosen *by* Model 1, so scoring Model 1 on
them is circular. The unbiased comparison repeats the pipeline on a random
sample of the prompts rather than the harvested subset:

```bash
python replication/validate.py --input data/google_local_prompts.csv \
    --limit 500 --ballots results/ballots_random500.jsonl
python replication/agreement.py --ballots results/ballots_random500.jsonl \
    --out results/validated_random500.csv
python replication/score_validated.py \
    --validated results/validated_random500.csv \
    --out results/table06_random.csv
```

---

## Results in this repository

| File | Contents |
|---|---|
| `replication/results/table01…table10*.csv/.tex` | Every reproduced table |
| `replication/results/table05c_amazon_ab.csv` | The Amazon text-field comparison |
| `results/ballots_*.jsonl` | Every individual judge response, raw |
| `results/validated_round3.csv` | 667 adversarial sentences with panel gold |
| `results/validated_random500.csv` | 449 random sentences with panel gold |
| `results/table06_new_data.csv`, `table06_random.csv` | Model scores on the new data |
| `results/calibration_report.txt` | Panel agreement against human gold |
| `results/amazon_contentonly_m*.txt` | Full external-corpus run logs |

The ballot files hold one JSON object per judge per sentence, including
the raw model output before parsing, so every gold label in this
repository can be traced back to the five responses that produced it.

---

## Notes

- `validate.py` and `agreement.py` judge four labels — Positive, Negative,
  No sentiment, Mixed sentiment — because the original crowdsourcing task
  offered four, not three. The ternary label is derived afterwards, and
  Mixed is excluded from the ternary set. The `label_distribution` field
  in the released DynaSent data retains a `mixed` key, which is what gave
  this away.
- `validate.py` is resumable. It keys on `(sentence id, judge)`, so an
  interrupted run can be restarted with the same command.
- All runs use fixed seeds and `temperature=0`.

