# Estimating Diversity

The goal of this repository is to estimate the diversity of usages per word in a given corpus. For this purpose, the repository contains the code required to compare two measures of polysemy of word occurrences in a text.

## The Two Methods for Estimating Diversity

The two methods are:

1. vMF: We fit and use a von Mises-Fisher distribution to estimate diversity, specifically the kappa parameter.
2. WiC-based: We are applying a transformer model trained on the word-in-context (WiC) task to distinguish whether words share a sense. We sample pairs of occurrences from the corpus and apply the WiC model to it. The mean probability that a pair differs in sense should tell us how diverse the senses of a target word in a corpus are.

## Simulation Study

To compare the two methods for estimating diversity, we use a simulation study with known ground-truth: We create artificial corpora where we know the correct senses by sampling from a WSD corpus. To make the corpus realistic, we start from a Zipfian distribution and vary two parameters: 

1. the slope of the Zipfian distribution, and 
2. the support of the Zipfian distribution, i.e. the number of senses.

We estimate the slope from the WSD corpus. (The estimate of the slope uses all occurrences of the lemma. In other processing steps, we require a minimum of 5 instances for simulation purposes, but the base slope should be naturalistic.)

By using sense-annotated occurrences to create corpora, we can compare the diversity estimates for datasets with known properties.

### Vocabularies

Our simulation study covers for each PoS up to 100 lemmata with at least 3 senses. If there are more than a 100 lemmata with 3+ senses, we select the 100 with most senses in the WSD corpus. For a sense to be considered in the count, we require a minimum of 5 instances. The filter of 5 is also applied later during the simulation. We find the lowest number of eligible lemmata, just 30, for adverbs .

The creation of these vocabs also creates statistic files (markdown format) which provide numbers for every PoS as well as an overview statistics file that describes the senses per lemma, instances per lemma, and instances per sense.

In addition, we have a list of 10 target verbs. However, 3 of these verbs have insufficient senses for being considered present in the source WSD dataset.

## Requirements

In addition to the libraries specified installed by the create-env recipe (conda + pip) in the justfile, this project requires the [Google WSD corpus](https://research.google/blog/a-large-corpus-for-supervised-word-sense-disambiguation/). These data are available at: github.com/google-research-datasets/word_sense_disambigation_corpora

The WiC data are available in the required format at:
- https://github.com/ameta13/mcl-wic/tree/main/data_dumped_full/wic_train-en-en
- https://github.com/cardiffnlp/TempoWiC/tree/main/data

The project also assumes the presence of a GPU.


## How to Run the Code

The commands required to run the code are provided in the justfile. The order of running the code is:

1. create-env: Create conda environment with required libraries.
2. create-vocab: Create the vocabularies of the lemmata with most senses (one file per PoS).
3. simulate-target-verbs: Simulate a corpus for the list of ten target verbs.
4. simulate-most-diverse-all: Simulate a corpus for each of the four most-diverse PoS vocabularies.
5. train-wic-tempowic: Train the WiC model (on WiC + TempoWiC) used for WiC-based scoring.
6. score-vmf-all: Score all simulated corpora with the vMF method, writing the results to `output/scores/<dataset>`.
7. score-wic-all: Score all simulated corpora with the WiC method, writing the results to `output/scores/<dataset>`.