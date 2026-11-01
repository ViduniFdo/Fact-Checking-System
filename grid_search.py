
from factchecking_main import read_passages, read_fact_examples
from factcheck import word_overlap_score

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = 'data/passages_bm25_ChatGPT_humfacts.jsonl'

def load_data():
    fact_to_passages = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(LABELS_PATH, fact_to_passages)
    labels = [1 if ex.label == "S" else 0 for ex in examples]
    return  examples, labels

def evaluate(examples,labels,threshold,remove_stop,use_stemming):
    correct = 0
    for ex, y in zip(examples, labels):
        score = word_overlap_score(ex.fact,ex.passages,remove_stop,use_stemming)
        pred = 1 if score >= threshold else 0
        correct += (pred == y)
    return correct/len(labels), correct

if __name__ == "__main__":
    examples, labels = load_data()
    print(f"Loaded {len(examples)} examples, {sum(labels)} supported")

    acc, correct = evaluate(examples,labels,threshold=0.75,remove_stop=True,use_stemming=True)
    print(f"Baseline: {correct}/{len(labels)} = {acc:.4f}")