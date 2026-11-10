import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from factcheck import EntailmentModel, EntailmentFactChecker
from factchecking_main import read_passages, read_fact_examples

def extract_model_errors():
    # 1. Load data
    passages_path = "data/passages_bm25_ChatGPT_humfacts.jsonl"
    labels_path = "data/dev_labeled_ChatGPT.jsonl"
    
    fact_to_passage_dict = read_passages(passages_path)
    examples = read_fact_examples(labels_path, fact_to_passage_dict)
    
    # 2. Instantiate entailment model
    model_name = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    
    # Set cuda=True if you are using GPU
    ent_model = EntailmentModel(model, tokenizer, cuda=False) 
    fact_checker = EntailmentFactChecker(ent_model)
    
    false_positives = [] # Gold = NS, Pred = S
    false_negatives = [] # Gold = S, Pred = NS
    
    print("Running error extraction...")
    for ex in examples:
        # Convert ground truth label (IR maps to NS)
        gold_label = "NS" if ex.label == "IR" else ex.label
        pred_label = fact_checker.predict(ex.fact, ex.passages)
        
        if gold_label == "NS" and pred_label == "S":
            false_positives.append({
                "fact": ex.fact,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "passages": [p["text"] for p in ex.passages]
            })
        elif gold_label == "S" and pred_label == "NS":
            false_negatives.append({
                "fact": ex.fact,
                "gold_label": gold_label,
                "pred_label": pred_label,
                "passages": [p["text"] for p in ex.passages]
            })
            
    # Save top 10 of each to a readable JSON file
    error_summary = {
        "false_positives": false_positives[:10],
        "false_negatives": false_negatives[:10]
    }
    
    with open("error_analysis_data.json", "w") as f:
        json.dump(error_summary, f, indent=4)
        
    print(f"Extraction complete! Found {len(false_positives)} False Positives and {len(false_negatives)} False Negatives.")
    print("Saved first 10 of each type to error_analysis_data.json")

if __name__ == "__main__":
    extract_model_errors()