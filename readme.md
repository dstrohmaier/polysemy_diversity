# Estimating Diversity

The goal of this repository is to estimate the diversity of usages per word in a given corpus. For this purpose, the repository contains the code required to compare two measures of polysemy of word occurrences in a text.

## The Two Methods for Estimating Diversity

The two methods are:

1. vMF-based: We fit and use a von Mises-Fisher (vMF) distribution to estimate diversity, specifically the we use the inverse of the $\kappa$ parameter as a prediction: $1 / \kappa$
2. WiC-based: We are applying a transformer model trained on the word-in-context (WiC) task to distinguish whether words share a sense.

### vMF Method

We explore two variations for the vMF-based score, both use the inverse of $\kappa$ but they differ in how they 

1. Inverse of $\kappa$ estimate from 
2. 

### WiC Method

The WiC model architecture and training are specified in [wic_models.md](./wic_models.md).

To create WiC-based diversity scores, we sample pairs of occurrences from the corpus and apply the WiC model to it. The mean probability that a pair differs in sense should tell us how diverse the senses of a target word in a corpus are.

## Evaluation: Polysemy-Based Simulation Study

To compare the two methods for estimating diversity, we use a simulation study with known ground-truth: We create artificial datasets where we know the correct senses by sampling from a WSD corpus. To make the datasets realistic, we start from a Zipfian distribution and vary two parameters:

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
| **1** | Shannon diversity | $\exp\!\left(-\sum_{i=1}^{R} p_i \ln p_i\right)$ | Each category weighted in proportion to its abundance |
| **2** | Simpson diversity | $1 / \sum_{i=1}^{R} p_i^{2}$                     | Weighted toward the most abundant categories          |

We will score for all three standard q values. Consequently, for each of the simulated datasets (lemmata) we will have three values with which we can correlate the scores. We use Spearman's Rank Correlation (SRC) for these.

Evenness could be operationalised as ${}^{1}D/{}^{0}D$ but we stick with the three diversity metrics.


#### Relation to the WiC-based Scoring

Let `p(diff)` be the probability that a random pair of usages for a word type differ in their sense as assigned by a WiC model. Assuming a perfect WiC model, the `p(diff)` is equivalent to the common formulation of Gini's diversity index: $1- \sum_i p^2_i$, where $p^2_i$ is the probability that two draws fall into category $i$. This index is dominatd by frequent senses and, therefore, provides a measure of the dominance of the most frequent sense. As can be seen, Gini's diversity index is closely related Simpson diversity, i.e. the the diversity score with $q=0$


### Vocabularies

Our simulation study covers for each PoS up to 100 lemmata with at least 3 senses. If there are more than a 100 lemmata with 3+ senses, we select the 100 with most senses in the WSD corpus. For a sense to be considered in the count, we require a minimum of 5 instances. The filter of 5 is also applied later during the simulation. We find the lowest number of eligible lemmata, just 30, for adverbs .

The creation of these vocabs also creates statistic files (markdown format) which provide numbers for every PoS as well as an overview statistics file that describes the senses per lemma, instances per lemma, and instances per sense.

In addition, we have a list of 10 target verbs. However, 3 of these verbs have insufficient senses for being considered present in the source WSD dataset.


## Evaluation:


## Requirements

In addition to the libraries specified installed by the create-env recipe (conda + pip) in the justfile, this project requires the [Google WSD corpus](https://research.google/blog/a-large-corpus-for-supervised-word-sense-disambiguation/). These data are available at: https://github.com/google-research-datasets/word_sense_disambigation_corpora

The primary WiC data are available in the required format at:
- https://github.com/ameta13/mcl-wic/tree/main/data_dumped_full/wic_train-en-en

In addition, we synthesise a WiC dataset from the FEWS WSD corpus:
- https://nlp.cs.washington.edu/fews/
The source of the FEWS corpus is

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

- Giulianelli, M., Del Tredici, M., & Fernández, R. (2020). Analysing Lexical Semantic Change with Contextualised Word Representations. Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics, 3960–3973. https://doi.org/10.18653/v1/2020.acl-main.365
- Schlechtweg, D., Cassotti, P., Noble, B., Alfter, D., Schulte im Walde, S., & Tahmasebi, N. (2024). More DWUGs: Extending and Evaluating Word Usage Graph Datasets in Multiple Languages. In Y. Al-Onaizan, M. Bansal, & Y.-N. Chen (Eds), Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing (pp. 14379–14393). Association for Computational Linguistics. https://doi.org/10.18653/v1/2024.emnlp-main.796
- Schlechtweg, D., McGillivray, B., Hengchen, S., Dubossarsky, H., & Tahmasebi, N. (2020). SemEval-2020 Task 1: Unsupervised Lexical Semantic Change Detection. In A. Herbelot, X. Zhu, A. Palmer, N. Schneider, J. May, & E. Shutova (Eds), Proceedings of the Fourteenth Workshop on Semantic Evaluation (pp. 1–23). International Committee for Computational Linguistics. https://doi.org/10.18653/v1/2020.semeval-1.1
- Schlechtweg, D., Schulte im Walde, S., & Eckmann, S. (2018). Diachronic Usage Relatedness (DURel): A Framework for the Annotation of Lexical Semantic Change. In M. Walker, H. Ji, & A. Stent (Eds), Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers) (pp. 169–174). Association for Computational Linguistics. https://doi.org/10.18653/v1/N18-2027
- Schlechtweg, D., Tahmasebi, N., Hengchen, S., Dubossarsky, H., & McGillivray, B. (2021). DWUG: A large Resource of Diachronic Word Usage Graphs in Four Languages. In M.-F. Moens, X. Huang, L. Specia, & S. W. Yih (Eds), Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing (pp. 7079–7091). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.emnlp-main.567


### NLP WiC 

-Hayashi, Y. (2025). Evaluating LLMs’ Capability to Identify Lexical Semantic Equivalence: Probing with the Word-in-Context Task. In O. Rambow, L. Wanner, M. Apidianaki, H. Al-Khalifa, B. D. Eugenio, & S. Schockaert (Eds), Proceedings of the 31st International Conference on Computational Linguistics (pp. 6985–6998). Association for Computational Linguistics. https://aclanthology.org/2025.coling-main.466/
- Pilehvar, M. T., & Camacho-Collados, J. (2019). WiC: The Word-in-Context Dataset for Evaluating Context-Sensitive Meaning Representations. In J. Burstein, C. Doran, & T. Solorio (Eds), Proceedings of the 2019 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 1 (Long and Short Papers) (pp. 1267–1273). Association for Computational Linguistics. https://doi.org/10.18653/v1/N19-1128

### vMF 

- 


[^1]: For this use of diversity measures, see [this blogpost](https://biostatsquid.com/hill-numbers/) by biostatsquid.
