"""
Lists every error of the final entailment system (run from the repo root): false positives (gold NS/IR, predicted S)
and false negatives (gold S, predicted NS), each with the highest-scoring premise and the 3-class probabilities.
"""
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from factchecking_main import read_passages, read_fact_examples
from factcheck import EntailmentModel, EntailmentFactChecker

MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

if __name__ == "__main__":
    examples = read_fact_examples("data/dev_labeled_ChatGPT.jsonl",
                                  read_passages("data/passages_bm25_ChatGPT_humfacts.jsonl"))
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME).eval()
    checker = EntailmentFactChecker(EntailmentModel(model, AutoTokenizer.from_pretrained(MODEL_NAME)))
    errors = {"FP": [], "FN": []}
    for ex in examples:
        premises = checker.candidate_premises(ex.fact, ex.passages)
        probs = checker.ent_model.check_entailment_batch(premises, ex.fact) if premises else []
        best = max(zip(probs, premises), key=lambda t: t[0][0]) if premises else None
        pred = "S" if best and best[0][0] >= checker.ent_threshold else "NS"
        gold = "S" if ex.label == "S" else "NS"
        if pred != gold:
            errors["FP" if pred == "S" else "FN"].append((ex, best))
    for kind, items in errors.items():
        print(f"\n===== {kind}: {len(items)} =====")
        for i, (ex, best) in enumerate(items, 1):
            print(f"\n[{kind}{i}] gold={ex.label} title={ex.passages[0]['title']}\nFACT: {ex.fact}")
            if best:
                print(f"probs(ent,neu,con)={[round(x, 2) for x in best[0]]}\nPREMISE: {best[1][:600]}")
            else:
                print("all premises pruned")
