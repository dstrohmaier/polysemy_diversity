# WiC Model

This document further specifies the WiC model.

## Model Details

| Property       | Value                          |
|----------------|--------------------------------|
| Backbone       | `answerdotai/ModernBERT-large` |
| Input format   | `lemma: sent1 sent2`           |
| Pooling format | `[u; v; \|u-v\|]`              |
| Token wrapping | `[unused0] target [unused1]`   |

### Pooling Format

[Conneau et al. (2017)](https://aclanthology.org/D17-1070.pdf) used the pooling format `[u; v; |u−v|; u⊙v]`. For their architecture (SentenceBERT), [Reimers & Gurevych (2019)](https://aclanthology.org/D19-1410.pdf) found that `u⊙v` reduced performance, so we dropped it.


### Token Wrapping

We wrap the target tokens in both sentences in boundary markers. These markers are available in the modernBERT vocabulary but unused, so we train our own special tokens.

## Dataset

The primary WiC data are available in the required format at:
- https://github.com/ameta13/mcl-wic/tree/main/data_dumped_full/wic_train-en-en

In addition, we synthesise a WiC dataset from the FEWS WSD corpus:
- https://nlp.cs.washington.edu/fews/

I also considered the [TempoWiC](https://github.com/cardiffnlp/TempoWiC/) dataset, but initiall exploration suggests it is too different.


### Synthetic Data from FEWS

The synthetic data are created to have a 50/50 split between positive and negative labels, i.e. half the word token pairs are of the same sense and the other half differ.

FEWS targets fewshot and zeroshot WSD, which leads it to often include an insufficient number of instances for a sense for our purpose. This particularly concerns the dev-split, which has no repetition at all. Hence, we create the validation data out of the train split of FEWS. We do not use the dev or test split at all.
