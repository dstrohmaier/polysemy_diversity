"""Shared thresholds for selecting lemmata and senses from the WSD data.

These govern which lemmata and senses are eligible -- both when building the
vocabularies (``create_vocabs.py``) and when simulating corpora
(``simulate_data.py`` / ``simulation.corpus_simulation``) -- so they live in one
place rather than being duplicated as separate defaults.
"""

# Minimum distinct sentences a sense needs to be usable. A sense seen fewer times is
# too sparse to simulate a corpus from or to count toward a lemma's diversity.
MIN_EXAMPLES = 5

# Minimum distinct senses a lemma needs to be included in a vocabulary. With fewer
# senses there is no diversity to vary, so the lemma is not a useful simulation target.
MIN_SENSES = 3
