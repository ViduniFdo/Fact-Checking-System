
from factchecking_main import read_passages, read_fact_examples

LABELS_PATH = "data/dev_labeled_ChatGPT.jsonl"
PASSAGES_PATH = 'data/passages_bm25_ChatGPT_humfacts.jsonl'

def load_data():
    fact_to_passages = read_passages(PASSAGES_PATH)
    examples = read_fact_examples(LABELS_PATH, fact_to_passages)
    labels = [1 if ex.label == "S" else 0 for ex in examples]
    return  examples, labels

if __name__ == "__main__":
    examples, labels = load_data()
    print(f"Loaded {len(examples)} examples, {sum(labels)} supported")
