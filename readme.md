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

We estimate the slope from the WSD corpus. 

By using sense-annotated occurrences to create corpora, we can compare the diversity estimates for datasets with known properties.

### Vocabularies

Our simulation study covers for each PoS the 100 lemmata with most senses in WordNet. In addition, we have a list of 10 target verbs.

## Requirements

In addition to the libraries specified installed by the create-env recipe (conda + pip) in the justfile, this project requires the [Google WSD corpus](https://research.google/blog/a-large-corpus-for-supervised-word-sense-disambiguation/).

The project also requires the presence of a GPU.


## How to Run the Code

The commands required to run the code are provided in the justfile. The order of running the code is:

1. create-env: Create conda environment with required libraries.
2. create-vocab: Create the vocabularies of the lemmata with most senses (one file per PoS).
3. simulate-target-verbs: Simulate a corpus for the list of ten target verbs.
4. simulate-most-diverse-all: Simulate a corpus for each of the four most-diverse PoS vocabularies.
5. train-wic-tempowic: Train the WiC model (on WiC + TempoWiC) used for WiC-based scoring.
6. score-vmf-all: Score all simulated corpora with the vMF method, writing the results to `output/scores/<dataset>`.
7. score-wic-all: Score all simulated corpora with the WiC method, writing the results to `output/scores/<dataset>`.