# Estimating Shifts in Diversity

The goal of this repository is to compare corpora and estimate the shift in the diversity of usages per word. For this purpose, the repository contains the code required to compare three measures of the shift (1 baseline + 2 methods) in polysemy of word occurrences in a text.

## One Baseline and Two Methods for Estimating Diversity

The baseline and the two methods are:

0. Cosine baseline: We use leave-one-out cosine distance to measure the diversity and then take a log-ratio to estimate shift in diversity.
1. vMF-based: We fit and use a von Mises-Fisher (vMF) distribution to estimate diversity, using the log-ratio of kappa parameters to indicate the shift in diversity.
2. WiC-based: We are applying a transformer model trained on the word-in-context (WiC) task to distinguish whether words share a sense, again taking a log-ratio to indicate shifts in diversity.


### Cosine Baseline

For the cosine baseline, we use leave-one-out (LOO) centroid distance as the diversity measure in a log-ratio for the source (S) and target corpus (T). That is, for each corpus, we compare each vector to the mean direction of every vector (centroid) except itself. Using the LOO-setup avoids the bias of comparing the vector against itself. The log-ratio is calculated as follows:

$$\log \frac{D_{\text{cos}}(T)}{D_{\text{cos}}(S)}$$

where $D_{\text{cos}}$ is the LOO cosine distance averaged over all vectors in the corpus.


### vMF Method

Our vMF method closely follows the work of Nagata et al. (2023) with a twist for easier readability. We take the log-ratio of the estimated $\kappa$ parameters of the von Mises-Fisher distribution for the source (S) and target corpus (T):

$$\log \frac{\kappa_S}{\kappa_T}$$

The $\kappa$ parameters grow _inversely_ with the diversity of the usages. We flip the ratio originally put forward by Nagata et al., i.e. put $\kappa_T$ in the denominator and $\kappa_S$ in the numerator, so that the log-ratio increases when the target corpus is more diverse.

We are assessing on smaller $n$ than Nagata et al. Our experiments show whether their variance-based method can also be applied to smaller corpora.

### WiC Method

The WiC method approximates the following method:

$$\log \frac{p(\text{same}|S)}{p(\text{same}|T)}$$

Effectively, we use the log of the ratio of two Simpson-diversity measures (see the discussions below). To see this, note that $p(\text{same}) \approx \sum_i p^2_i$, where $i$ indicates the senses available. The right-hand side is the inverse of the Simpson-diversity. This equivalence also allows us to derive our score as follows:

$$\frac{\text{Simpson-diversity}_T}{\text{Simpson-diversity}_S} = \frac{\frac{1}{\sum_{i=1}^{R} p_{T,i}^{2}}}{\frac{1}{ \sum_{i=1}^{R} p_{S,i}^{2}}} = \frac{\sum_{i=1}^{R} p_{S,i}^{2}}{\sum_{i=1}^{R} p_{T,i}^{2}}$$

This ratio increases with the diversity of the target corpus being larger than that of the source corpus. This is the same direction as for the vMF method.

The WiC model architecture and training are specified in [wic_models.md](./wic_models.md).

To create WiC-based diversity scores, we sample pairs of occurrences from the corpus and apply the WiC model to it. The mean probability that a pair differs in sense should tell us how diverse the senses of a target word in a corpus are.

## First Evaluation: Polysemy-Based Simulation Study

To compare the two methods for estimating shifts in diversity, we use a simulation study with known ground-truth: We create artificial datasets where we know the correct senses by sampling from a WSD corpus. The approach is inspired by [Schlechtweg & Walde (2020)](https://arxiv.org/abs/2001.03216), but their simulation is targeting the emergence and disappearance of lexical senses over time change. We are more interested in the fine-grained control of individual corpora on which we can then compare different methods targeting shifts in diversity.

To make the datasets realistic, we start from a Zipfian distribution and vary two parameters:

1. **Richness**: the support of the Zipfian distribution, i.e. richness as the number of distinct senses.
2. **Evenness**: the slope of the Zipfian distribution (evenness of sense distribution), and


We estimate the slope from the WSD corpus. (The estimate of the slope uses all occurrences of the lemma. In other processing steps, we require a minimum of 5 instances for simulation purposes, but the base slope should be naturalistic.)

By using sense-annotated occurrences to create dataset with simulated distributions, we can compare the diversity estimates for datasets with known properties.

To interpret this simulation-based evaluation, we have to consider a few key facts:

1. The task is close the original WiC task, which arguably favours the WiC-based scoring. Therefore, one should always compare against the results of our second evaluation task.
2. The simulated data vary along two dimensions (evenness and richness) but both the vMF and the WiC-based method produce a single score. Part of our goal is to see to which degree the two methods track the two different dimensions. Therefore, we provide scoring along both dimensions.


### Source and Target Corpus

The source and target corpus are defined per word. The primary source corpus is the one with the steepest Zipfian slope-parameter and lowest number of senses $k$. We make the following comparisons:

- We compare all other corpora against this primary corpus. 
- We compare along the dimension of $k$, i.e. we compare the corpora that differ only in $k$ while sharing slope.
- We compare along the dimension of slope, i.e. we compare the corpora that differ only in slope while sharing senses $k$.

For all these comparisons, the source corpus is always the one with the expected lower diversity (steeper slope or lower $k$).

Different sizes for source and target corpus lead to problems. To address this issue, we always sample the larger corpus down to the size of the smaller.

### Multi-Dimension Scoring

To score the diversity of usages along the two dimensions of **richness** and **evenness** , it is helpful to look at the general form of common diversity scores:[^1]

$${}^{q}D = \left( \sum_{i=1}^{R} p_i^{q} \right)^{1/(1-q)}$$

| q     | Name              | Value it reduces to                              | How it weights categories                             |
|-------|-------------------|--------------------------------------------------|-------------------------------------------------------|
| **0** | Richness          | $R$                                              | All present categories count equally                  |
| **1** | Shannon diversity | $\exp\left(-\sum_{i=1}^{R} p_i \ln p_i\right)$ | Each category weighted in proportion to its abundance |
| **2** | Simpson diversity | $1 / \sum_{i=1}^{R} p_i^{2}$                     | Weighted toward the most abundant categories          |

Setting $q=0$, the scores track exactly **richness**. With $q\in \{1,2 \}$, the scores track a combination of richness and evenness. To operationalise **evenness**, we use $E = {}^{1}D/{}^{0}D$, that is the ratio between Shannon diversisty and richness.

We will score the ground-truth values for all fource metrics and then also consider the log-ratios for the source-target corpus comparisons:

$$\log \frac{{}^{q}D(T)}{{}^{q}D(S)}\, \text{ for } q \in \{0,1,2\}$$

and 

$$\log \frac{E(T)}{E(S)}$$


Consequently, for each of the simulated datasets (lemmata) we will have four shift values with which we can correlate the scores of the baseline and the two methods. We use Spearman's Rank Correlation (SRC) for these.


#### Relation to the WiC-based Scoring

Let `p(same)` be the probability that a random pair of word tokens of the same type share a sense. Assuming a perfect WiC model, the $p(\text{same}) \approx \sum_i p^2_i$, where $p^2_i$ is the probability that two draws fall into category $i$. That is, `p(same)` is the inversion of the Simpson diversity index. This index is dominated by frequent senses and, therefore, provides a measure of the dominance of the most frequent sense.

As a consequence, we should consider the WiC-scoring for $q=2$ as a check on the performance of the underlying WiC-model, which is different from how we should interpret the other approaches on this task. 


#### Qualification of the Eveness Metric

The evenness metric would be invariant to the number of senses ($k$) if we held the distribution constant while adding senses. However, our simulation does not achieve that because we use the same Zipfian slope for different $k$, which effectively shifts the distribution. As a result, $E$ falls as $k$ rises at a fixed slope, but that just indicates a downside of our simulation which relies on Zipf truncation.


### Vocabularies

Our simulation study covers for each PoS up to 100 lemmata with at least 3 senses. If there are more than 100 lemmata with 3+ senses, we select the 100 with most senses in the WSD corpus. For a sense to be considered in the count, we require a minimum of 5 instances. The filter of 5 is also applied later during the simulation. We find the lowest number of eligible lemmata, just 30, for adverbs. For such small numbers, we expect not separability between the methods.

The creation of these vocabs also creates statistic files (markdown format) which provide numbers for every PoS as well as an overview statistics file that describes the senses per lemma, instances per lemma, and instances per sense.


## Second Evaluation: Diachronic Data Study

For the second evaluation we use the DWUG EN dataset. This dataset of historical word use provides information that can be used to estimate both **richness** and **evenness**. We use the two decade groupings:

1. 1810–1860
2. 1960–2010

As in the first evaluation
- we calculate the ground truth values of the log-ratio of the diversity measures (using grouping 1 as the source and grouping 2 as the target corpus), and
- we evaluate the vMF- and the WiC-based method using Spearman's rank correlation.

vMF- and WiC-based scores are calculated on the DWUG usages. Note that we switch from a setup in which the source corpus is the least diverse one to a setup in which the source corpus is the older one.

We evaluate per lemma and apply the same downsample rule as in the first evaluation to make sure the corpora are of the same size.

The DWUG EN study likely favours WiC less because it is not based on a dictionary-style sense inventory as is commonly used to conceptualise the WiC task.

### Senses and Ground Truth in DWUG

The sense inventory is DWUG's optimal clustering (`clusters/opt/`). Usages that clustering leaves unassigned (cluster `-1`) are dropped from both the ground truth and the data the scorers see. Dropping them reproduces the cluster frequency distributions DWUG itself publishes in `stats/opt/stats_groupings.csv`.

The ground truth is computed per grouping over **all** its (noise-free) usages, before the downsample. Downsampling exists only to stop a corpus-size difference contaminating the *scores*; the diversity of a grouping is a property of the data, not of whichever subsample a scorer happened to see.

One asymmetry between the two evaluations is worth keeping in mind when reading the results side by side. In the simulation, the ground-truth sense distribution is the *design* distribution: a property of the generative process, independent of the sample drawn from it. In DWUG it is the *observed* distribution of roughly 100 usages per grouping. Richness (q=0) is the order most exposed to this: it is downward-biased at finite sample size, and the bias is uneven across lemmata, since a few groupings are considerably smaller than the rest after noise removal. The Shannon and Simpson orders (q=1, 2) weight by abundance and are correspondingly more robust.

(Possible extension: Use a gradual ground truth. The current sketch uses the clustering discretization but we can use the raw gradual data and our scores are gradual anyway.)

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
6. score-cosine-all: Score all simulation datasets with the cosine baseline, writing per-pair log-ratios to `output/scores/<dataset>`.
7. score-vmf-all: Score all simulation datasets with the vMF method (per-pair log-ratios).
8. score-wic-all: Score all simulation datasets with the WiC method (per-pair log-ratios).
9. analyse-comparative-all: Correlate each method's shift score against the ground-truth diversity shifts (richness/evenness), writing tables and figures to `output/analysis/<dataset>`.

Steps 2–4 and 6–9 cover the first evaluation. The second evaluation reuses the same scoring and analysis code on the DWUG data, and needs the WiC model from step 5:

10. prepare-dwug: Split DWUG EN into per-lemma decade-grouping corpora (`g1` = 1810–1860, `g2` = 1960–2010) under `source_data/dwug_corpora`, writing a preparation report to `source_data/dwug_corpora/tables/dwug_preparation.csv`.
11. score-dwug-all: Score the DWUG corpora with all three methods into `output/scores/dwug_en`.
12. analyse-comparative-dwug: Correlate the DWUG shift scores against the DWUG ground truth, writing to `output/analysis/dwug_en`.

Each scoring step compares corpus *pairs* per lemma (source vs. target) and writes a `<method>_pair_scores.csv`; the comparative analysis joins these against the ground-truth `log(qD(T)/qD(S))` shifts for q ∈ {0,1,2}.

The two evaluations differ only in how pairs are enumerated and where the ground-truth sense distribution comes from. Both are selected with `--dataset` (`simulated` or `dwug`) on `score_data.py` and `run_analysis.py`; the simulated grid yields three comparison schemes per lemma, DWUG a single `diachronic` pair.

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
- Kishino, Yamagiwa, Nagata, Yokoi, & Shimodaira (2025). [Quantifying Lexical Semantic Shift via Unbalanced Optimal Transport](https://doi.org/10.18653/v1/2025.acl-long.774). *ACL 2025*, 15913–15933.
- Nagata, Takamura, Otani, & Kawasaki (2023). [Variance Matters: Detecting Semantic Differences without Corpus/Word Alignment](https://doi.org/10.18653/v1/2023.emnlp-main.965). *EMNLP 2023*, 15609–15622.


### Hill-Measures and Evenness

- Hill (1973). [Diversity and Evenness: A Unifying Notation and Its Consequences](https://doi.org/10.2307/1934352). *Ecology*, 54(2), 427–432.
- Tuomisto (2012). [An Updated Consumer's Guide to Evenness and Related Indices](https://doi.org/10.1111/j.1600-0706.2011.19897.x). *Oikos*, 121(8), 1203–1218.


### Other Metrics

- Hoffman, Lambon Ralph, & Rogers (2013). [Semantic Diversity: A Measure of Semantic Ambiguity Based on Variability in the Contextual Usage of Words](https://doi.org/10.3758/s13428-012-0278-x). *Behavior Research Methods*, 45(3), 718–730.

### Embeddings and Models

- Ethayarajh (2019). [How Contextual are Contextualized Word Representations? Comparing the Geometry of BERT, ELMo, and GPT-2 Embeddings](https://doi.org/10.18653/v1/D19-1006). *EMNLP-IJCNLP 2019*, 55–65.

### Other Datasets

- Blevins, Joshi, & Zettlemoyer (2021). [FEWS: Large-Scale, Low-Shot Word Sense Disambiguation with the Dictionary](https://doi.org/10.18653/v1/2021.eacl-main.36). *EACL 2021*, 455–465.

[^1]: For this use of diversity measures, see [this blogpost](https://biostatsquid.com/hill-numbers/) by biostatsquid.
