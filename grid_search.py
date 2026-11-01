
import csv
from itertools import product

from factchecking_main import read_passages, read_fact_examples
from factcheck import (preprocess_text,get_passage_words,count_overlapping_words,jaccard_similarity_score,tfidf_similarity_score,rouge_similarity_score,)


LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = 'data/passages_bm25_ChatGPT_humfacts.jsonl'

def load_data():
    fact_to_passages = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(LABELS_PATH, fact_to_passages)
    labels = [1 if ex.label == "S" else 0 for ex in examples]
    return  examples, labels

def recall_score(fact, passages, remove_stop, use_stemming):
    fact_words = set(preprocess_text(fact, remove_stop, use_stemming))
    if not fact_words:
        return 0.0
    passage_words = get_passage_words(passages, remove_stop, use_stemming)
    return count_overlapping_words(fact_words, passage_words) / len(fact_words)


METRICS = {
    "word_recall": (recall_score, True),
    "jaccard": (jaccard_similarity_score, True),
    "tfidf_cosine": (tfidf_similarity_score, True),
    "rouge1_fmeasure": (lambda  f, p, s, t: rouge_similarity_score(f, p, "rouge1", "fmeasure"), False),
    "rouge1_recall": (lambda f, p, s, t: rouge_similarity_score(f, p, "rouge1", "recall"), False),
    "rouge2_recall": (lambda f, p, s, t: rouge_similarity_score(f, p, "rouge2", "recall"), False),
    "rougeL_recall": (lambda f, p, s, t: rouge_similarity_score(f, p, "rougeL", "recall"), False),
}


def best_threshold(scores, labels):
    best_acc, best_t = 0.0, 0.0
    for t in sorted(set(scores)):
        correct = sum((s >= t) == bool(y) for s, y in zip(scores, labels))
        acc = correct / len(labels)
        if acc > best_acc:
            best_acc, best_t = acc, t
    return  best_acc, best_t

def main():
    examples, labels = load_data()
    print(f"Loaded {len(examples)} examples, {sum(labels)} supported")

    results = []
    for name, (func, uses_prep) in METRICS.items():
        combos = list(product([True, False], [True, False])) if uses_prep else [(None, None)]
        for remove_stop, use_stemming in combos:
            scores = [func(ex.fact, ex.passages, remove_stop, use_stemming) for ex in examples]
            acc, thr = best_threshold(scores, labels)
            results.append((name, remove_stop, use_stemming, round(acc, 4), round(thr, 4)))
            print(f"{name:16} stop={remove_stop!s:5} stem= {use_stemming!s:5} "
                  f"acc={acc:.4f} thr={thr:.4f}")
    results.sort(key=lambda  r: r[3], reverse=True)
    with open("grid_search_results.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "remove_stopwords", "stemming", "accuracy", "best_threshold"])
        writer.writerows(results)

    print("\nTop 5 configurations:")
    for r in results[:5]:
        print(r)



if __name__ == "__main__":
    main()

