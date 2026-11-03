"""
Entailment experiments (run from the repo root). Scores every title-prefixed window of WINDOW consecutive passage
sentences whose word recall of the fact is at least 0.1 with the NLI model, caches the probabilities, then sweeps the
word-overlap pruning threshold, the cap on premises and the entailment threshold offline.
Usage: python tune_entailment.py [window]   (window 1 = single sentences, 3 = final system)
Results are written to entailment_tuning_results_window<N>.csv.
"""
import csv
import json
import os
import sys

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factchecking_main import read_passages, read_fact_examples
from factcheck import EntailmentModel, preprocess_text, count_overlapping_words, split_into_sentences

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = "data/passages_bm25_ChatGPT_humfacts.jsonl"
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
MIN_RECALL = 0.1


def window_probs(examples, window):
    """ Per example: list of (word recall of the fact, [P_ent, P_neu, P_con]) for each candidate premise window """
    cache_path = f"entailment_probs_window{window}_title.json"
    if os.path.exists(cache_path):
        return json.load(open(cache_path))
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    ent_model = EntailmentModel(model, tokenizer, False)
    out = []
    for ex in tqdm(examples):
        fact_words = set(preprocess_text(ex.fact, True, True))
        premises, seen = [], set()
        for p in ex.passages:
            sents = split_into_sentences(p["text"])
            for i in range(max(1, len(sents) - window + 1)):
                chunk = " ".join(sents[i:i + window])
                if not chunk or chunk in seen:
                    continue
                seen.add(chunk)
                recall = count_overlapping_words(fact_words, set(preprocess_text(chunk, True, True))) / max(len(fact_words), 1)
                if recall >= MIN_RECALL:
                    premises.append((recall, p["title"] + ". " + chunk))
        probs = ent_model.check_entailment_batch([t for _, t in premises], ex.fact) if premises else []
        out.append([(r, pr) for (r, _), pr in zip(premises, probs)])
    json.dump(out, open(cache_path, "w"))
    return out


def evaluate(pairs, labels, prune, cap, thr):
    """ Accuracy when keeping premises with recall >= prune (best cap by recall) and predicting S if max P_ent >= thr """
    correct = 0
    for cand, y in zip(pairs, labels):
        kept = sorted([c for c in cand if c[0] >= prune], key=lambda c: -c[0])[:cap]
        score = max((p[0] for _, p in kept), default=0.0)
        correct += (score >= thr) == bool(y)
    return correct / len(labels)


if __name__ == "__main__":
    window = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    examples = read_fact_examples(LABELS_PATH, read_passages(PASSAGES_PATH))
    labels = [1 if ex.label == "S" else 0 for ex in examples]
    pairs = window_probs(examples, window)
    rows = []
    for prune in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
        for cap in [3, 5, 8, 1000]:
            for thr in [i / 20 for i in range(1, 20)]:
                rows.append((prune, cap, thr, round(evaluate(pairs, labels, prune, cap, thr), 4)))
    rows.sort(key=lambda r: -r[3])
    with open(f"entailment_tuning_results_window{window}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["prune_threshold", "max_premises", "entailment_threshold", "accuracy"])
        w.writerows(rows)
    for r in rows[:10]:
        print(r)
