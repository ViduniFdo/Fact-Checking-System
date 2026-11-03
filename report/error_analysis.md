---
title: "CM3604 Deep Learning – Part 3: Error Analysis of the Entailment Fact-Checker"
geometry: margin=2.2cm
fontsize: 11pt
---

## 1. System and setup

The system scores each fact against Wikipedia passages with DeBERTa-v3-base (MNLI/FEVER/ANLI). Premises are windows of 3 consecutive sentences prefixed with the page title. They are pruned to those sharing at least 40% of the fact's (stemmed, stop-word-free) words, keeping the best 8. The fact is predicted *Supported* if the maximum entailment probability over premises is at least 0.4. This gives **86.4% accuracy (191/221)**: 10 false positives (FP) and 20 false negatives (FN). Sentence-level premises without the title scored 81.0%, and word-overlap-only scored 80.5%. I examined all 10 FPs and the first 10 FNs (FN1–FN10). The remaining 10 FNs are used only as a check on the categories (§3).

## 2. Fine-grained error categories

**FP-A: Plausible-but-unstated inference.** The premise supports a related, weaker or more specific claim (synonym, hypernym, location, or world knowledge), and the model treats the fact as entailed although the gold label is NS/IR.

**FP-B: Entity mention mistaken for the relation.** The fact's entity appears in the premise, but in a different role (e.g. as the source of a degree, or as a broadcaster named in passing). High lexical overlap makes the model entail a relation (*worked at*, *is an X*) the text never states.

**FN-A: Composite or evaluative claims.** The fact bundles several pieces of information or is vague/evaluative ("best known for", "first handheld…", "popular", "numerous"). No single premise states it, so the model answers *neutral*.

**FN-B: Value or reference conflict.** The premise contains a close but different value (date, duration), or the fact has an unresolved pronoun, so the model answers *contradiction* even though the gold label is S.

**FN-C: Evidence not surfaced.** The supporting text is not in the best-scoring premise, either because word-overlap pruning removed it (the fact uses different words from the passage) or because the passage says it differently.

## 3. Aggregate statistics

| Category | Count (examined 10 + 10) | Examples | Count over all 20 FNs |
|---|---|---|---|
| FP-A Plausible-but-unstated inference | 7 / 10 FP | FP1, 2, 3, 5, 8, 9, 10 | – |
| FP-B Entity mention, not relation | 3 / 10 FP | FP4, 6, 7 | – |
| FN-A Composite/evaluative claim | 3 / 10 FN | FN1, 2, 3 | 10 / 20 |
| FN-B Value/reference conflict | 3 / 10 FN | FN4, 5, 6 | 4 / 20 |
| FN-C Evidence not surfaced | 4 / 10 FN | FN7, 8, 9, 10 | 6 / 20 |

Errors are concentrated in few documents. 12 of 20 FNs come from two biographies (Ji Sung and Jonathan Haagensen), and the generic "X is popular / has starred in numerous…" facts alone give 7 FNs. Pruning directly caused 3 of the 20 FNs (FN9, FN11 and FN12 have no premise left), but it cut the number of premises scored from 5,261 single sentences to 975 windows at no accuracy cost.

## 4. Detailed examples

**Example 1 (FP-A).** *Fact:* "Nisar Ahmad Kakru was Kashmiri." *Premise (excerpt):* "…(born 26 October 1949 Baramulla, Kashmir, India) was the Chief Justice of the High Court of Andhra Pradesh…" *Gold:* NS. *Predicted:* S (P(entail) = 0.99). *Type:* FP-A. Being born in Kashmir makes "Kashmiri" likely, but it is not stated, and the person served in Andhra Pradesh. The model accepts a plausible world-knowledge leap, whereas the annotators required explicit support.

**Example 2 (FP-B).** *Fact:* "He has worked at Mayo Clinic." *Premise:* "Oliphant has a Ph.D. in Biomedical Engineering from the Mayo Clinic and B.S. and M.S. degrees … from Brigham Young University. Oliphant was an Assistant Professor … at Brigham Young University…" *Gold:* NS. *Predicted:* S (0.85). *Type:* FP-B. The Mayo Clinic is mentioned only as the place of a degree, and all employment in the text is at BYU. The model matches the entity and a nearby "he has … Mayo Clinic" pattern rather than checking the relation.

**Example 3 (FN-B).** *Fact:* "He served for many years as a pastor." *Premise:* "Between 1986 and 1989, Malcolm a pastor of the Jerusalem Christian Assembly. … Malcolm served from 1991 – 2000 as Chairman of 'Christian Action for Israel'…" *Gold:* S. *Predicted:* NS (P(contradiction) = 0.82). *Type:* FN-B. The fact's "many years" and pronoun "He" do not align with the 3-year pastorate in this window. The model reads a mismatch in duration and role as contradiction, and does not combine it with other evidence in the passage.

## 5. Reflection and improvements

The two error directions have different causes. FPs come from over-trusting lexical and topical closeness, and FNs come from claims that need aggregation, or whose evidence is split across passages or worded differently. Possible improvements: (i) require margin over the neutral/contradiction probabilities and not only a threshold on entailment, since most FPs are very confident (P > 0.85) and would need a stricter, calibrated decision rule; (ii) decompose composite facts such as "best known for…" and aggregate evidence over several premises; (iii) replace overlap-only pruning with a softer retriever (embeddings), to rescue FN-C cases such as "Rio de Janeiro is in Brazil"; (iv) resolve pronouns ("He", "They") with the page title before inference. The dev set is small (221 facts, 30 errors), and all thresholds were tuned on it, so the 86.4% figure is probably optimistic.
