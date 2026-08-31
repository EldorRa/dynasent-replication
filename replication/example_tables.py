import pandas as pd

from common import R1, R2, load_dataset, save_table

PAPER_TABLE3 = {
    "Good food nasty attitude by hostesses .",
    "Not much of a cocktail menu that I saw.",
    "I scheduled the work for 3 weeks later.",
    "I was very mistaken, it was much more!",
    "It is a gimmick, but when in Rome, I get it.",
    "Probably a little pricey for lunch.",
    "But this is strictly just my opinion.",
    "The price was okay, not too pricey.",
    "The only downside was service was a little slow.",
    "However there is a 2 hr seating time limit.",
    "With Alex, I never got that feeling.",
    "Its ran very well by management.",
}

PAPER_TABLE8 = {
    "The place was somewhat good and not well",
    "I bought a new car and met with an accident.",
    "The retail store is closed for now at least.",
    "Prices are basically like garage sale prices.",
    "That book was good. I need to get rid of it.",
    "I REALLY wanted to like this place",
    "But I'm going to leave my money for the next vet.",
    "once upon a time the model made a super decision.",
    "I cook my caribbean food and it was okay",
    "This concept is really cool in name only.",
    "Wow, it'd be super cool if you could join us",
    "Knife cut thru it like butter! It was great.",
}


def format_responses(dist):
    return ", ".join(
        sorted(k[:3] for k, v in dist.items() for _ in range(len(v))))


def sample_short_examples(split, model_col, minlen=30, maxlen=50,
                          random_state=1):
    df = pd.DataFrame(split)
    df = df[df.gold_label.notnull()]
    df = df[(df.sentence.str.len() > minlen) & (df.sentence.str.len() < maxlen)]
    picks = [grp.sample(1, random_state=random_state)
             for _, grp in df.groupby(["gold_label", model_col])]
    samp = pd.concat(picks)
    samp["responses"] = samp.label_distribution.apply(format_responses)
    samp = samp[["sentence", model_col, "gold_label", "responses"]]
    return samp.sort_values([model_col, "gold_label"]).reset_index(drop=True)


def report(table, title, split_path, model_col, paper_sentences):
    samp = sample_short_examples(load_dataset(split_path), model_col)
    print(f"\n{table}: {title}  ({len(samp)} rows)")
    print(samp.to_string(index=False))
    save_table(samp, table)

    lengths = samp.sentence.str.len()
    norm = lambda s: " ".join(s.split())
    paper_norm = {norm(s) for s in paper_sentences}
    overlap = sum(1 for s in samp.sentence if norm(s) in paper_norm)

    print(f"\n  rows: {len(samp)}")
    print(f"  all sentences 30-50 chars: "
          f"{bool((lengths > 30).all() and (lengths < 50).all())}")
    print(f"  sentences also in the published table: {overlap}/{len(samp)}")


if __name__ == "__main__":
    report("table03", "Round 1 examples", R1["train"], "model_0_label",
           PAPER_TABLE3)
    report("table08", "Round 2 examples", R2["train"], "model_1_label",
           PAPER_TABLE8)
