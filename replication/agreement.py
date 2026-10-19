import argparse
import collections
import csv
import json
import os

LABELS = ("positive", "negative", "no sentiment", "mixed sentiment")
TERNARY = {"positive": "positive",
           "negative": "negative",
           "no sentiment": "neutral"}
TERNARY_LABELS = ("positive", "negative", "neutral")


def load(path):
    rows = collections.defaultdict(dict)
    meta = {}
    with open(path, encoding="utf8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            rows[r["id"]][r["judge"]] = r.get("label")
            meta[r["id"]] = r
    return rows, meta


def fleiss_kappa(ballots, categories):
    counts = []
    for labels in ballots:
        if any(l is None for l in labels):
            continue
        c = collections.Counter(labels)
        counts.append([c.get(l, 0) for l in categories])
    if not counts:
        return None, 0
    n = sum(counts[0])
    if n < 2:
        return None, len(counts)
    N = len(counts)
    p = [sum(row[j] for row in counts) / (N * n) for j in range(len(categories))]
    P = [(sum(x * x for x in row) - n) / (n * (n - 1)) for row in counts]
    P_bar = sum(P) / N
    P_e = sum(x * x for x in p)
    if P_e >= 1:
        return None, N
    return (P_bar - P_e) / (1 - P_e), N


def majority(labels, n_judges):
    valid = [l for l in labels if l is not None]
    if not valid:
        return None
    top, count = collections.Counter(valid).most_common(1)[0]
    return top if count * 2 > n_judges else None


def f1_scores(gold, pred, categories=TERNARY_LABELS):
    out = {}
    for lab in categories:
        tp = sum(1 for g, p in zip(gold, pred) if g == lab and p == lab)
        fp = sum(1 for g, p in zip(gold, pred) if g != lab and p == lab)
        fn = sum(1 for g, p in zip(gold, pred) if g == lab and p != lab)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        out[lab] = 200 * prec * rec / (prec + rec) if prec + rec else 0.0
    out["macro"] = sum(out[l] for l in categories) / len(categories)
    if set(categories) == set(TERNARY_LABELS):
        out["neutral_gap"] = (out["positive"] + out["negative"]) / 2 - out["neutral"]
    return out


def main(a):
    rows, meta = load(a.ballots)
    judges = sorted({j for v in rows.values() for j in v})
    n_judges = len(judges)

    ballots, ids = [], []
    for sid, per_judge in rows.items():
        if len(per_judge) == n_judges:
            ballots.append([per_judge[j] for j in judges])
            ids.append(sid)

    print(f"{len(rows):,} sentences, {n_judges} judges")
    print(f"{len(ballots):,} with a complete ballot")

    print("\nparse failure rate by judge")
    for j in judges:
        bad = sum(1 for v in rows.values() if v.get(j) is None)
        print(f"  {j:16} {bad:5,} / {len(rows):,}  ({100*bad/max(len(rows),1):.1f}%)")

    print("\nresponse distribution by judge")
    print("  " + "judge".ljust(16) + "".join(f"{l[:9]:>12}" for l in LABELS))
    for j in judges:
        c = collections.Counter(v.get(j) for v in rows.values())
        print("  " + j.ljust(16) + "".join(f"{c.get(l,0):12,}" for l in LABELS))

    k4, used4 = fleiss_kappa(ballots, LABELS)
    print(f"\nFleiss' kappa, 4 categories, {used4:,} ballots: "
          f"{'n/a' if k4 is None else f'{k4:.3f}'}")

    tern = [[TERNARY.get(l) if l else None for l in b] for b in ballots]
    k3, used3 = fleiss_kappa([b for b in tern if all(b)], TERNARY_LABELS)
    print(f"Fleiss' kappa, 3 categories, {used3:,} ballots: "
          f"{'n/a' if k3 is None else f'{k3:.3f}'}")
    print("  paper reports 0.616 (Round 1) and 0.681 (Round 2)")

    maj4 = {sid: majority(b, n_judges) for sid, b in zip(ids, ballots)}
    resolved4 = {s: m for s, m in maj4.items() if m}
    print(f"\nmajority reached on {len(resolved4):,} of {len(ballots):,} "
          f"({100*len(resolved4)/max(len(ballots),1):.1f}%)")
    for lab, c in collections.Counter(resolved4.values()).most_common():
        print(f"  {lab:18} {c:5,}")

    gold3 = {s: TERNARY[m] for s, m in resolved4.items() if m in TERNARY}
    n_mixed = sum(1 for m in resolved4.values() if m == "mixed sentiment")
    print(f"\n{len(gold3):,} sentences have a ternary gold label")
    print(f"{n_mixed:,} resolved to Mixed sentiment and are excluded from the "
          f"ternary set")

    cal = [s for s in gold3 if meta[s].get("gold")]
    if cal:
        g = [meta[s]["gold"] for s in cal]
        p = [gold3[s] for s in cal]
        acc = 100 * sum(x == y for x, y in zip(g, p)) / len(g)
        f1 = f1_scores(g, p)
        print(f"\nCALIBRATION against {len(cal):,} known human labels")
        print(f"  panel accuracy     {acc:.1f}%")
        for lab in TERNARY_LABELS:
            print(f"  F1 {lab:9}       {f1[lab]:.1f}")
        print(f"  macro F1           {f1['macro']:.1f}")
        print("\n  confusion (rows = human gold, cols = panel)")
        print("            " + "".join(f"{l[:4]:>9}" for l in TERNARY_LABELS))
        for gi in TERNARY_LABELS:
            row = [sum(1 for x, y in zip(g, p) if x == gi and y == pj)
                   for pj in TERNARY_LABELS]
            print(f"  {gi:9} " + "".join(f"{v:9,}" for v in row))

    star = [s for s in gold3 if meta[s].get("star_label")]
    if star:
        ok = sum(1 for s in star if meta[s]["star_label"] == gold3[s])
        print(f"\npanel vs the star-derived label, {len(star):,} sentences")
        print(f"  agree    {ok:,} ({100*ok/len(star):.1f}%)")
        print(f"  disagree {len(star)-ok:,} ({100*(len(star)-ok)/len(star):.1f}%)")
        by = collections.defaultdict(lambda: [0, 0])
        for s in star:
            r = meta[s].get("review_rating")
            by[r][1] += 1
            if meta[s]["star_label"] != gold3[s]:
                by[r][0] += 1
        print("\n  disagreement by star rating")
        for r in sorted(by, key=lambda x: str(x)):
            d, t = by[r]
            print(f"    {str(r):8} {d:5,} / {t:5,}  ({100*d/t:.1f}%)")

    if a.out:
        os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
        with open(a.out, "w", encoding="utf8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "sentence", "gold_label", "panel_label_4way",
                        "star_label", "review_rating"] + judges)
            for sid, b in zip(ids, ballots):
                if sid not in gold3:
                    continue
                m = meta[sid]
                w.writerow([sid, m["sentence"], gold3[sid], maj4[sid],
                            m.get("star_label"), m.get("review_rating")] + b)
        print(f"\nwrote {a.out}  ({len(gold3):,} validated sentences)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ballots", required=True)
    p.add_argument("--out", default="")
    main(p.parse_args())
