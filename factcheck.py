# factcheck.py

import torch
from typing import List
import numpy as np
import spacy
import gc
import re
from nltk.stem import PorterStemmer
import math
from collections import Counter
from rouge_score import rouge_scorer


class FactExample(object):
    """
    :param fact: A string representing the fact to make a prediction on
    :param passages: List[dict], where each dict has keys "title" and "text". "title" denotes the title of the
    Wikipedia page it was taken from; you generally don't need to use this. "text" is a chunk of text, which may or
    may not align with sensible paragraph or sentence boundaries
    :param label: S, NS, or IR for Supported, Not Supported, or Irrelevant. Note that we will ignore the Irrelevant
    label for prediction, so your model should just predict S or NS, but we leave it here so you can look at the
    raw data.
    """
    def __init__(self, fact: str, passages: List[dict], label: str):
        self.fact = fact
        self.passages = passages
        self.label = label

    def __repr__(self):
        return repr("fact=" + repr(self.fact) + "; label=" + repr(self.label) + "; passages=" + repr(self.passages))


class EntailmentModel(object):
    def __init__(self, model, tokenizer, cuda=False):
        self.model = model
        self.tokenizer = tokenizer
        self.cuda = cuda

    def check_entailment(self, premise: str, hypothesis: str):
        with torch.no_grad():
            # Tokenize the premise and hypothesis
            inputs = self.tokenizer(premise, hypothesis, return_tensors='pt', truncation=True, padding=True)
            if self.cuda:
                inputs = {key: value.to('cuda') for key, value in inputs.items()}
            # Get the model's prediction
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Note that the labels are ["entailment", "neutral", "contradiction"]. There are a number of ways to map
        # these logits or probabilities to classification decisions; you'll have to decide how you want to do this.

        raise Exception("Not implemented")

        # To prevent out-of-memory (OOM) issues during autograding, we explicitly delete
        # objects inputs, outputs, logits, and any results that are no longer needed after the computation.
        del inputs, outputs, logits
        gc.collect()

        # return something


class FactChecker(object):
    """
    Fact checker base type
    """

    def predict(self, fact: str, passages: List[dict]) -> str:
        """
        Makes a prediction on the given sentence
        :param fact: same as FactExample
        :param passages: same as FactExample
        :return: "S" (supported) or "NS" (not supported)
        """
        raise Exception("Don't call me, call my subclasses")


class RandomGuessFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        prediction = np.random.choice(["S", "NS"])
        return prediction


class AlwaysEntailedFactChecker(FactChecker):
    def predict(self, fact: str, passages: List[dict]) -> str:
        return "S"

stop_words = ["a", "an", "the", "and", "or", "but", "of", "in", "on", "at", "to", "for", "from", "by", "with", "as",
            "is", "are", "was", "were", "be", "been", "being", "am", "it", "its", "this", "that", "these", "those",
            "he", "she", "they", "his", "her", "their", "him", "them", "who", "whom", "which", "what", "when",
            "where", "while", "also", "has", "have", "had", "do", "does", "did", "not", "no", "so", "than",
            "then", "there", "into", "about", "over", "after", "before", "during", "between", "both", "each", "other",
            "such", "can", "could", "would", "should", "may", "might", "will", "shall", "i", "we", "you", "your", "our"
              ]

stemmer = PorterStemmer()

def tokenize_text(text:str) -> List[str]:
    text = text.lower()
    text = text.replace("<s>", " ")
    text = text.replace("</s>", " ")
    words = re.findall(r"[a-z0-9]+", text)
    return words

def remove_stop_words(words: List[str]) -> List[str]:
    kept_words = []
    for word in words:
        if word not in stop_words:
            kept_words.append(word)
    return kept_words

def stem_words(words: List[str]) -> List[str]:
    stemmed_words = []
    for word in words:
        stemmed_words.append(stemmer.stem(word))
    return stemmed_words

def preprocess_text(text: str, remove_stop: bool = True, use_stemming: bool = True) -> List[str]:
    words = tokenize_text(text)
    if remove_stop:
        words = remove_stop_words(words)
    if use_stemming:
        words = stem_words(words)
    return words

def get_passage_words(passages: List[dict], remove_stop: bool = True, use_stemming: bool = True) -> set:
    passage_words = set()
    for passage in passages:
        words = preprocess_text(passage["text"], remove_stop, use_stemming)
        for word in words:
            passage_words.add(word)
    return passage_words

def count_overlapping_words(fact_words: set, passage_words: set) -> int:
    overlap_count = 0
    for word in fact_words:
        if word in passage_words:
            overlap_count = overlap_count + 1
    return overlap_count


def jaccard_similarity_score(fact: str, passages: List[dict], remove_stop: bool = True, use_stemming: bool = True) -> float:
    """ Computes Jaccard Similarity: |Fact ∩ Passages| / |Fact ∪ Passages| """
    fact_words = set(preprocess_text(fact, remove_stop, use_stemming))
    if not fact_words:
        return 0.0

    passage_words = get_passage_words(passages, remove_stop, use_stemming)
    if not passage_words:
        return 0.0

    intersection = len(fact_words.intersection(passage_words))
    union = len(fact_words.union(passage_words))
    return intersection / union if union > 0 else 0.0


def tfidf_similarity_score(fact: str, passages: List[dict], remove_stop: bool = True, use_stemming: bool = True) -> float:
    """ Computes Cosine Similarity between TF-IDF vectors of Fact and Passages """
    fact_tokens = preprocess_text(fact, remove_stop, use_stemming)
    passage_tokens = []
    for p in passages:
        passage_tokens.extend(preprocess_text(p["text"], remove_stop, use_stemming))

    if not fact_tokens or not passage_tokens:
        return 0.0

    vocab = set(fact_tokens).union(set(passage_tokens))
    doc_count = 2

    tf_fact = Counter(fact_tokens)
    tf_passage = Counter(passage_tokens)

    vec_fact, vec_passage = [], []
    for term in vocab:
        df = (1 if term in tf_fact else 0) + (1 if term in tf_passage else 0)
        idf = math.log((1 + doc_count) / (1 + df)) + 1
        vec_fact.append((tf_fact[term] / len(fact_tokens)) * idf)
        vec_passage.append((tf_passage[term] / len(passage_tokens)) * idf)

    dot_product = sum(f * p for f, p in zip(vec_fact, vec_passage))
    norm_fact = math.sqrt(sum(f ** 2 for f in vec_fact))
    norm_passage = math.sqrt(sum(p ** 2 for p in vec_passage))

    if norm_fact == 0 or norm_passage == 0:
        return 0.0
    return dot_product / (norm_fact * norm_passage)


def rouge_similarity_score(fact: str, passages: List[dict], metric: str = "rouge1", stat: str = "fmeasure") -> float:
    """ Computes maximum ROUGE score across passages """
    if not passages:
        return 0.0

    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    max_score = 0.0
    for p in passages:
        scores = scorer.score(target=fact, prediction=p["text"])
        score_val = getattr(scores[metric], stat)
        if score_val > max_score:
            max_score = score_val
    return max_score

# Set this variable to control which metric runs: "word_recall", "jaccard", "tfidf", or "rouge"
# Set active_metric to: "word_recall", "jaccard", "tfidf", or "rouge"
active_metric = "word_recall"

def word_overlap_score(fact: str, passages: List[dict], remove_stop: bool = True, use_stemming: bool = True) -> float:
    if active_metric == "word_recall":
        fact_words = set(preprocess_text(fact, remove_stop, use_stemming))
        if len(fact_words) == 0:
            return 0.0
        passage_words = get_passage_words(passages, remove_stop, use_stemming)
        return count_overlapping_words(fact_words, passage_words) / len(fact_words)

    elif active_metric == "jaccard":
        return jaccard_similarity_score(fact, passages, remove_stop, use_stemming)

    elif active_metric == "tfidf":
        return tfidf_similarity_score(fact, passages, remove_stop, use_stemming)

    elif active_metric == "rouge":
        return rouge_similarity_score(fact, passages, metric="rouge1", stat="fmeasure")
    
    return 0.0


class WordRecallThresholdFactChecker(FactChecker):
    def __init__(self, threshold: float = 0.75, remove_stop: bool = True, use_stemming: bool = True):
        self.default_threshold = threshold
        self.remove_stop = remove_stop
        self.use_stemming = use_stemming

    def predict(self, fact: str, passages: List[dict]) -> str:
        # Dynamically set threshold based on the selected metric
        if active_metric == "jaccard":
            thresh = 0.015
        elif active_metric == "tfidf":
            thresh = 0.08
        elif active_metric == "rouge":
            thresh = 0.05
        else:
            thresh = self.default_threshold

        score = word_overlap_score(fact, passages, self.remove_stop, self.use_stemming)
        if score >= thresh:
            return "S"
        else:
            return "NS"

class EntailmentFactChecker(FactChecker):
    def __init__(self, ent_model):
        self.ent_model = ent_model

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")


# OPTIONAL
class DependencyRecallThresholdFactChecker(FactChecker):
    def __init__(self):
        self.nlp = spacy.load('en_core_web_sm')

    def predict(self, fact: str, passages: List[dict]) -> str:
        raise Exception("Implement me")

    def get_dependencies(self, sent: str):
        """
        Returns a set of relevant dependencies from sent
        :param sent: The sentence to extract dependencies from
        :param nlp: The spaCy model to run
        :return: A set of dependency relations as tuples (head, label, child) where the head and child are lemmatized
        if they are verbs. This is filtered from the entire set of dependencies to reflect ones that are most
        semantically meaningful for this kind of fact-checking
        """
        # Runs the spaCy tagger
        processed_sent = self.nlp(sent)
        relations = set()
        for token in processed_sent:
            ignore_dep = ['punct', 'ROOT', 'root', 'det', 'case', 'aux', 'auxpass', 'dep', 'cop', 'mark']
            if token.is_punct or token.dep_ in ignore_dep:
                continue
            # Simplify the relation to its basic form (root verb form for verbs)
            head = token.head.lemma_ if token.head.pos_ == 'VERB' else token.head.text
            dependent = token.lemma_ if token.pos_ == 'VERB' else token.text
            relation = (head, token.dep_, dependent)
            relations.add(relation)
        return relations