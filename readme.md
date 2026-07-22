# Estimating Diversity

The goal of this repository is to estimate the diversity of usages per word in a given corpus. For this purpose, the repository contains the code required to compare two measures of polysemy of word occurrences in a text.

## The Two Methods for Estimating Diversity

The two methods are:

1. vMF-based: We fit and use a von Mises-Fisher (vMF) distribution to estimate diversity.
2. WiC-based: We are applying a transformer model trained on the word-in-context (WiC) task to distinguish whether words share a sense.

### vMF Method

TO BE SPECIFIED

### WiC Method

The WiC model architecture and training are specified in [wic_models.md](./wic_models.md).

To create WiC-based diversity scores, we sample pairs of occurrences from the corpus and apply the WiC model to it. The mean probability that a pair differs in sense should tell us how diverse the senses of a target word in a corpus are.

## Evaluation: Polysemy-Based Simulation Study

To compare the two methods for estimating diversity, we use a simulation study with known ground-truth: We create artificial datasets where we know the correct senses by sampling from a WSD corpus. The approach is similar to that by [Schlechtweg & Walde (2020)](https://arxiv.org/abs/2001.03216).

To make the datasets realistic, we start from a Zipfian distribution and vary two parameters:

1. **Evenness**: the slope of the Zipfian distribution (eveness of sense distribution), and
2. **Richness**: the support of the Zipfian distribution, i.e. richness as the the number of distinct senses.

We estimate the slope from the WSD corpus. (The estimate of the slope uses all occurrences of the lemma. In other processing steps, we require a minimum of 5 instances for simulation purposes, but the base slope should be naturalistic.)

By using sense-annotated occurrences to create dataset with simulated distributions, we can compare the diversity estimates for datasets with known properties.

To intepret this simulation-based evaluation, we have to consider a few key facts:

1. The task is close the original WiC task, which arguably favours the WiC-based scoring. Therefore, one should always compare against the results of our second evaluation task.
2. The simulated data vary along two dimensions (evenness and richness) but both the vMF and WiC-based method produce a single score. Part of our goal is to see to which degree the two methods track the two different dimensions. Therefore, we provide scoring along both dimensions.


### Multi-Dimension Scoring

To score the diversity of usages along the two dimensions of **evenness** and **richness**, it is helpful to look at the general form of common diversity scores:[^1]

$${}^{q}D = \left( \sum_{i=1}^{R} p_i^{q} \right)^{1/(1-q)}$$

| q     | Name              | Value it reduces to                              | How it weights categories                             |
|-------|-------------------|--------------------------------------------------|-------------------------------------------------------|
| **0** | Richness          | $R$                                              | All present categories count equally                  |
| **1** | Shannon diversity | $\exp\left(-\sum_{i=1}^{R} p_i \ln p_i\right)$ | Each category weighted in proportion to its abundance |
| **2** | Simpson diversity | $1 / \sum_{i=1}^{R} p_i^{2}$                     | Weighted toward the most abundant categories          |

We will score for all three standard q values. Consequently, for each of the simulated datasets (lemmata) we will have three values with which we can correlate the scores. We use Spearman's Rank Correlation (SRC) for these.

Evenness could be operationalised as ${}^{1}D/{}^{0}D$ but we stick with the three diversity metrics.


#### Relation to the WiC-based Scoring

Let `p(diff)` be the probability that a random pair of usages for a word type differ in their sense as assigned by a WiC model. Assuming a perfect WiC model, the `p(diff)` is equivalent to the common formulation of Gini's diversity index: $1- \sum_i p^2_i$, where $p^2_i$ is the probability that two draws fall into category $i$. This index is dominatd by frequent senses and, therefore, provides a measure of the dominance of the most frequent sense. As can be seen, Gini's diversity index is closely related Simpson diversity, i.e. the diversity score with $q=2$


### Vocabularies

Our simulation study covers for each PoS up to 100 lemmata with at least 3 senses. If there are more than a 100 lemmata with 3+ senses, we select the 100 with most senses in the WSD corpus. For a sense to be considered in the count, we require a minimum of 5 instances. The filter of 5 is also applied later during the simulation. We find the lowest number of eligible lemmata, just 30, for adverbs .

The creation of these vocabs also creates statistic files (markdown format) which provide numbers for every PoS as well as an overview statistics file that describes the senses per lemma, instances per lemma, and instances per sense.

In addition, we have a list of 10 target verbs. However, 3 of these verbs have insufficient senses for being considered present in the source WSD dataset.


## Evaluation: Diachronic Data Study

Decision open, plausible candidates include DWUG EN.


## Requirements

In addition to the libraries specified installed by the create-env recipe (conda + pip) in the justfile, this project requires the [Google WSD corpus](https://research.google/blog/a-large-corpus-for-supervised-word-sense-disambiguation/). These data are available at: https://github.com/google-research-datasets/word_sense_disambigation_corpora

The primary WiC data are available in the required format at:
- https://github.com/ameta13/mcl-wic/tree/main/data_dumped_full/wic_train-en-en

In addition, we synthesise a WiC dataset from the FEWS WSD corpus:
- https://nlp.cs.washington.edu/fews/

The source of the FEWS corpus is the Wikitionary data. 

The project also assumes the presence of a GPU.


## How to Run the Code

The commands required to run the code are provided in the justfile. The order of running the code is:


1. create-env: Create conda environment with required libraries.
2. create-vocab: Create the vocabularies of the lemmata with most senses (one file per PoS).
3. simulate-target-verbs: Create a dataset with simulated sense distribution for the list of ten target verbs.
4. simulate-most-diverse-all: Create datasets with simulated sense distributions for each of the four most-diverse PoS vocabularies.
5. train-wic-fews: Train the WiC model (on WiC + synthetic FEWS) used for WiC-based scoring.
6. score-vmf-all: Score all simulation datasets with the vMF method, writing the results to `output/scores/<dataset>`.
7. score-wic-all: Score all simulation datasets with the WiC method, writing the results to `output/scores/<dataset>`.


## Relevant Literature

### NLP Semantic Change

- Giulianelli, Del Tredici, & Fernández (2020). [Analysing Lexical Semantic Change with Contextualised Word Representations](https://doi.org/10.18653/v1/2020.acl-main.365). *ACL 2020*, 3960–3973.
- Schlechtweg, Cassotti, Noble, Alfter, Schulte im Walde, & Tahmasebi (2024). [More DWUGs: Extending and Evaluating Word Usage Graph Datasets in Multiple Languages](https://doi.org/10.18653/v1/2024.emnlp-main.796). *EMNLP 2024*, 14379–14393.
- Schlechtweg, McGillivray, Hengchen, Dubossarsky, & Tahmasebi (2020). [SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection](https://doi.org/10.18653/v1/2020.semeval-1.1). *SemEval 2020*, 1–23.
- Schlechtweg, Schulte im Walde, & Eckmann (2018). [Diachronic Usage Relatedness (DURel): A Framework for the Annotation of Lexical Semantic Change](https://doi.org/10.18653/v1/N18-2027). *NAACL-HLT 2018*, 169–174.
- Schlechtweg & Schulte im Walde (2020). [Simulating Lexical Semantic Change from Sense-Annotated Data](https://doi.org/10.48550/arXiv.2001.03216). *arXiv:2001.03216*.
- Schlechtweg, Tahmasebi, Hengchen, Dubossarsky, & McGillivray (2021). [DWUG: A Large Resource of Diachronic Word Usage Graphs in Four Languages](https://doi.org/10.18653/v1/2021.emnlp-main.567). *EMNLP 2021*, 7079–7091.

### NLP WiC

- Hayashi (2025). [Evaluating LLMs' Capability to Identify Lexical Semantic Equivalence: Probing with the Word-in-Context Task](https://aclanthology.org/2025.coling-main.466/). *COLING 2025*, 6985–6998.
- Pilehvar & Camacho-Collados (2019). [WiC: The Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations](https://doi.org/10.18653/v1/N19-1128). *NAACL-HLT 2019*, 1267–1273.

### vMF

- Banerjee, Dhillon, Ghosh, & Sra (2005). [Clustering on the Unit Hypersphere using von Mises-Fisher Distributions](https://jmlr.org/papers/v6/banerjee05a.html). *JMLR*, 6(46), 1345–1382.
- Nagata, Takamura, Otani, & Kawasaki (2023). [Variance Matters: Detecting Semantic Differences without Corpus/Word Alignment](https://doi.org/10.18653/v1/2023.emnlp-main.965). *EMNLP 2023*, 15609–15622.

### Embeddings and Models

- Ethayarajh (2019). [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings](https://doi.org/10.18653/v1/D19-1006). *EMNLP-IJCNLP 2019*, 55–65.

### Other Datasets

- Blevins, Joshi, & Zettlemoyer (2021). [FEWS: Large-Scale, Low-Shot Word Sense Disambiguation with the Dictionary](https://doi.org/10.18653/v1/2021.eacl-main.36). *EACL 2021*, 455–465.

[^1]: For this use of diversity measures, see [this blogpost](https://biostatsquid.com/hill-numbers/) by biostatsquid.
