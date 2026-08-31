import json
import os
import sys
from collections import Counter

from common import DATA_DIR, R1, REPO_ROOT

EXPECTED_LINES = {
    "dynasent-v1.1-round01-yelp-train.jsonl": 94459,
    "dynasent-v1.1-round01-yelp-dev.jsonl": 3600,
    "dynasent-v1.1-round01-yelp-test.jsonl": 3600,
    "dynasent-v1.1-round02-dynabench-train.jsonl": 18535,
    "dynasent-v1.1-round02-dynabench-dev.jsonl": 720,
    "dynasent-v1.1-round02-dynabench-test.jsonl": 720,
    "sst-dev-validated.jsonl": 1101,
}

ok = True


def check(label, condition, detail=""):
    global ok
    mark = "PASS" if condition else "FAIL"
    if not condition:
        ok = False
    print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))


def note(label, state, detail=""):
    print(f"  [{state}] {label}" + (f"  {detail}" if detail else ""))


print("1. Environment")
print(f"  python {sys.version.split()[0]}")
for pkg in ("numpy", "pandas", "sklearn", "statsmodels", "torch",
            "transformers", "matplotlib"):
    try:
        m = __import__(pkg)
        print(f"  {pkg:<13} {getattr(m, '__version__', '?')}")
    except ImportError:
        print(f"  {pkg:<13} NOT INSTALLED")

print("\n2. Dataset files")
check("dynasent-v1.1/ exists", os.path.isdir(DATA_DIR), DATA_DIR)
if os.path.isdir(DATA_DIR):
    for fname, expected in EXPECTED_LINES.items():
        path = os.path.join(DATA_DIR, fname)
        if not os.path.exists(path):
            check(fname, False, "missing")
            continue
        with open(path) as f:
            n = sum(1 for _ in f)
        check(fname, n == expected, f"{n:,} lines (expected {expected:,})")

print("\n3. gold_label == 3/5 majority?")
if os.path.exists(R1["dev"]):
    mismatches = 0
    with open(R1["dev"]) as f:
        for line in f:
            d = json.loads(line)
            counts = Counter({k: len(v)
                              for k, v in d["label_distribution"].items()})
            top, n = counts.most_common(1)[0]
            recomputed = top if n >= 3 else None
            if recomputed != d["gold_label"]:
                mismatches += 1
    check("Round 1 dev gold labels recompute", mismatches == 0,
          f"{mismatches} mismatch(es)")

print("\n4. Models and compute")
try:
    import torch
    build = "CPU-only build" if "+cpu" in torch.__version__ else "CUDA build"
    if torch.cuda.is_available():
        note("GPU", "READY", torch.cuda.get_device_name(0))
    else:
        note("GPU", "----", f"torch {torch.__version__} is a {build}")
except ImportError:
    note("torch", "----", "not installed")

for name in ("dynasent_model0.bin", "dynasent_model1.bin"):
    path = os.path.join(REPO_ROOT, "models", name)
    size = os.path.getsize(path) / 1e6 if os.path.exists(path) else 0
    if size > 100:
        note(f"models/{name}", "READY", f"{size:.0f} MB")
    elif size:
        note(f"models/{name}", "BAD ", f"only {size:.0f} MB")
    else:
        note(f"models/{name}", "----", "not downloaded")

print()
print("Sections 1-3 all passed." if ok else "A check in sections 1-3 FAILED.")
sys.exit(0 if ok else 1)
