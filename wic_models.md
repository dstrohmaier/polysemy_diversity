# WiC Model

This document further specifies the WiC model.

## Model Details

| Property | Value |
|---|---|
| Backbone | `answerdotai/ModernBERT-large` |
| Input format | `lemma: sent1 sent2` |
| Pooling format | `[u; v; \|u-v\|]` |

### Pooling Format

https://aclanthology.org/D17-1070.pdf used the pooling format `[u; v; |u−v|; u⊙v]`. For their architecture (SentenceBert), https://aclanthology.org/D19-1410.pdf found the `u⊙v` reduced performance, so we dropped it.


## Dataset

