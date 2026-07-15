# WiC Model

This document further specifies the WiC model.

## Model Details

| Property            | Value                                                      |
|---------------------|------------------------------------------------------------|
| Architecture        | Target-vector classifier (`TargetVectorForWiC`)            |
| Backbone            | `answerdotai/ModernBERT-large`                             |
| Encoder input       | `"{lemma}: {sentence1}" [SEP] sentence2`, target-wrapped   |
| Target vectors      | masked mean-pool of the target subwords in each sentence   |
| Interaction feature | `[u; v; \|u-v\|]`                                          |
| Classifier head     | MLP: `Linear(3H→H) → GELU → Linear(H→2)` (dropout 0.1)     |
| Token wrapping      | `[unused0] target [unused1]`                               |

Both sentences are encoded together in one pass. Importantly, the model locates the target word in *each* sentence and extracts its own contextual vector — `u` from sentence 1, `v` from sentence 2. The target subwords are identified by the boundary markers (see [Token Wrapping](#token-wrapping)) and masked-mean-pooled. The InferSent-style interaction feature `[u; v; |u−v|]` is then fed to the MLP head. Implemented in `wic/target_vector_model.py`; the target masks are produced by `preprocess_wic_targets` in `wic/preprocessing.py`.

### Interaction Feature

[Conneau et al. (2017)](https://aclanthology.org/D17-1070.pdf) used the format `[u; v; |u−v|; u⊙v]`. For their architecture (SentenceBERT), [Reimers & Gurevych (2019)](https://aclanthology.org/D19-1410.pdf) found that `u⊙v` reduced performance, so we dropped it.


### Token Wrapping

We wrap the target tokens in both sentences in boundary markers. These markers are available in the modernBERT vocabulary but unused, so we train our own special tokens.

### Truncation Fall-back

Encoder inputs are truncated to `max_length=256` tokens. On long inputs this can push a target occurrence (or just its closing boundary marker) past the truncation point, leaving the span unterminated. When that happens for either sentence, we emit an **all-zero target mask** for that sentence rather than letting an open span bleed to the end of the sequence.

An all-zero mask means there are no target subwords to pool, so masked mean-pooling falls back to the **zero vector** for that sentence's `u` or `v` (`_pool_target` clamps the mask count to a minimum of 1 to avoid dividing by zero). The affected example is **kept**, not dropped; a per-batch count of such examples is logged as a warning during preprocessing. In practice this is rare given the 256-token limit, but it degrades those examples' features, so a spike in the warning count is a signal that inputs are running long. Implemented in `_target_mask_from_markers` (`wic/preprocessing.py`) and `_pool_target` (`wic/target_vector_model.py`).

## Dataset

The primary WiC data are available in the required format at:
- https://github.com/ameta13/mcl-wic/tree/main/data_dumped_full/wic_train-en-en

In addition, we synthesise a WiC dataset from the FEWS WSD corpus:
- https://nlp.cs.washington.edu/fews/

I also considered the [TempoWiC](https://github.com/cardiffnlp/TempoWiC/) dataset, but initial exploration suggests it is too different.


### Synthetic Data from FEWS

The synthetic data are created to have a 50/50 split between positive and negative labels, i.e. half the word token pairs are of the same sense and the other half differ.

FEWS targets fewshot and zeroshot WSD, which leads it to often include an insufficient number of instances for a sense for our purpose. This particularly concerns the dev-split, which has no repetition at all. Hence, we create the validation data out of the train split of FEWS. We do not use the dev or test split at all.
