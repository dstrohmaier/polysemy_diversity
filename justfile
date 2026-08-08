# The conda env's shared libraries must come first on the loader path for every
# recipe that runs project code. A non-existent path entry is silently skipped, so
# this is harmless during create-env (when ../diversity_env does not exist yet).

export LD_LIBRARY_PATH := "../diversity_env/lib:" + env("LD_LIBRARY_PATH", "")

# GPU used by every recipe that touches jax/cupy/torch. Override per invocation with
# `just gpu=0 <recipe> ...` rather than editing recipes one by one.
gpu := "1"

# Needed by torch.use_deterministic_algorithms (set in scoring's make_reproducible):
# cuBLAS reads it at CUDA init, too early for Python to set it.
export CUBLAS_WORKSPACE_CONFIG := ":4096:8"

unit-test:
    python -m unittest discover -s './tests' -p '*_tests.py'

create-env:
    #!/usr/bin/env bash
    conda create -p ../diversity_env just libstdcxx-ng -c conda-forge
    conda run -p ../diversity_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    conda run -p ../diversity_env pip install jax[cuda13] numpy pandas scipy scikit-learn seaborn tabulate datasets transformers tqdm evaluate accelerate nltk spacy
    conda run -p ../diversity_env python -m spacy download en_core_web_sm

# ---- Vocab Creation

create-vocab vocab="most_diverse" n="100":
    python create_vocabs.py source_data/vocabs {{ vocab }} source_data/word_sense_disambigation_corpora -n {{ n }}

# --- Simulation

simulate-target-verbs:
    python simulate_data.py source_data/word_sense_disambigation_corpora source_data/vocabs/target_verbs.json source_data/simulated_data/target_verbs

simulate-most-diverse pos $CUDA_VISIBLE_DEVICES=gpu:
    python simulate_data.py source_data/word_sense_disambigation_corpora source_data/vocabs/most_diverse_{{ pos }}.json source_data/simulated_data/most_diverse_{{ pos }}

simulate-most-diverse-all:
    #!/usr/bin/env bash
    for pos in noun verb adj adv; do
        just simulate-most-diverse "$pos"
    done

# --- DWUG preparation (second evaluation)

# Kept out of source_data/simulated_data so the simulation's *-all recipes, which
# glob that directory, do not pick DWUG up.

# Split DWUG EN into per-lemma decade-grouping corpora for the scorers.
prepare-dwug dwug_root="source_data/dwug_en" output_dir="source_data/dwug_corpora":
    python prepare_dwug.py {{ dwug_root }} {{ output_dir }}

# --- Training WiC

train model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES=gpu:
    python create_wic_model.py "{{ model }}" source_data output/models

train-fews model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES=gpu:
    python create_wic_model.py "{{ model }}" source_data output/models --dataset fews

train-wic-fews model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES=gpu:
    python create_wic_model.py "{{ model }}" source_data output/models --dataset wic+fews

predict_efcamdat model="answerdotai/ModernBERT-large" train_dataset="wic" $CUDA_VISIBLE_DEVICES=gpu:
    #!/usr/bin/env bash
    # Training stores the model name slash-replaced (model_name.replace("/", "--")).
    model_path="{{ model }}"
    model_path="${model_path//\//--}"
    python apply_wic.py "output/models/${model_path}/{{ train_dataset }}/final" source_data/efcamdat output/efcamdat/{{ train_dataset }}

# --- Scoring

score-wic sim_dir output_dir model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES=gpu:
    python score_data.py wic {{ sim_dir }} {{ output_dir }} --base-model "{{ model }}"

score-wic-all:
    #!/usr/bin/env bash
    for sim_dir in source_data/simulated_data/*/; do
        name=$(basename "$sim_dir")
        just score-wic "$sim_dir" "output/scores/$name"
    done

score-vmf sim_dir output_dir model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES=gpu:
    python score_data.py vmf {{ sim_dir }} {{ output_dir }} --hf-model-name "{{ model }}"

score-vmf-all:
    #!/usr/bin/env bash
    for sim_dir in source_data/simulated_data/*/; do
        name=$(basename "$sim_dir")
        just score-vmf "$sim_dir" "output/scores/$name"
    done

score-cosine sim_dir output_dir model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES=gpu:
    python score_data.py cosine {{ sim_dir }} {{ output_dir }} --hf-model-name "{{ model }}"

score-cosine-all:
    #!/usr/bin/env bash
    for sim_dir in source_data/simulated_data/*/; do
        name=$(basename "$sim_dir")
        just score-cosine "$sim_dir" "output/scores/$name"
    done

# --- Scoring (DWUG, second evaluation)

# --dataset dwug switches the pair enumerator to the single g1->g2 comparison
# per lemma.

# Score the prepared DWUG corpora with one method (cosine, vmf, or wic).
score-dwug method dwug_dir="source_data/dwug_corpora" output_dir="output/scores/dwug_en" model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES=gpu:
    #!/usr/bin/env bash
    set -euo pipefail
    if [ "{{ method }}" = "wic" ]; then
        python score_data.py wic {{ dwug_dir }} {{ output_dir }} --dataset dwug --base-model "{{ model }}"
    else
        python score_data.py {{ method }} {{ dwug_dir }} {{ output_dir }} --dataset dwug --hf-model-name "{{ model }}"
    fi

score-dwug-all dwug_dir="source_data/dwug_corpora" output_dir="output/scores/dwug_en":
    #!/usr/bin/env bash
    for method in cosine vmf wic; do
        just score-dwug "$method" "{{ dwug_dir }}" "{{ output_dir }}"
    done

# --- Analysis

analyse-raw-simulated sim_dir output_dir:
    python run_analysis.py raw_simulated {{ sim_dir }} {{ output_dir }}

analyse-raw-simulated-all:
    #!/usr/bin/env bash
    for sim_dir in source_data/simulated_data/*/; do
        name=$(basename "$sim_dir")
        just analyse-raw-simulated "$sim_dir" "output/analysis/$name"
    done

analyse-wic-simulated sim_dir output_dir:
    python run_analysis.py wic_simulated {{ sim_dir }} {{ output_dir }}

analyse-wic-simulated-all:
    #!/usr/bin/env bash
    for sim_dir in source_data/simulated_data/*/; do
        name=$(basename "$sim_dir")
        just analyse-wic-simulated "$sim_dir" "output/analysis/$name"
    done

analyse-comparative scores_dir sim_dir output_dir:
    python run_analysis.py comparative {{ scores_dir }} {{ output_dir }} {{ sim_dir }}

analyse-comparative-all:
    #!/usr/bin/env bash
    for sim_dir in source_data/simulated_data/*/; do
        name=$(basename "$sim_dir")
        just analyse-comparative "output/scores/$name" "$sim_dir" "output/analysis/$name"
    done

analyse-comparative-dwug scores_dir="output/scores/dwug_en" dwug_dir="source_data/dwug_corpora" output_dir="output/analysis/dwug_en":
    python run_analysis.py comparative {{ scores_dir }} {{ output_dir }} {{ dwug_dir }} --dataset dwug

# Globs on the output side (not simulated_data) so results of datasets whose sim
# dirs have since been removed are cleaned up too.
clean-comparative-all:
    rm -rf output/analysis/*/comparative

# --- Data Transfer

FLAMINGO := "ds858@flamingo.cl.cam.ac.uk:/local/scratch/ds858/wic_shift"

# Shared confirm-then-sync helper. `args` is forwarded verbatim to both the
# dry-run and the real rsync invocation, so it must fully specify source and
# destination, plus any --filter rules the caller needs. It is expanded unquoted,
# so paths must not contain spaces.
#
# rsync and ssh need nothing from the conda env, whose libtinfo makes bash warn
# about missing version information. Bash prints that at startup, before the
# script's first line, so the shebang clears the variable rather than the body.
_rsync_with_confirm *args:
    #!/usr/bin/env -S env -u LD_LIBRARY_PATH bash
    set -euo pipefail
    # Check the dry run's exit status separately: piping it straight into
    # `grep ... || true` would mask an rsync failure as "no deletions".
    if ! dry_run=$(rsync -rtvu --delete --dry-run --exclude-from=.rsyncignore {{args}}); then
        echo "Dry run failed; aborting before the real sync." >&2
        exit 1
    fi
    deletions=$(printf '%s\n' "$dry_run" | grep '^deleting ' || true)
    if [ -n "$deletions" ]; then
        echo "-- Dry run: files that would be deleted --"
        echo "$deletions"
        if [ ! -t 0 ]; then
            echo "Deletions pending but no terminal to confirm on; aborting." >&2
            exit 1
        fi
        read -rp "Proceed with sync? [y/N] " reply
        if [[ ! "$reply" =~ ^[Yy]([Ee][Ss])?$ ]]; then
            echo "Aborted."
            exit 1
        fi
    fi
    rsync -rtvu --delete --progress --exclude-from=.rsyncignore {{args}}


# Push the working tree to flamingo.
push2flamingo: (_rsync_with_confirm "./" FLAMINGO)

# Pull remote evaluation results and analysis outputs. Protects locally-held
# models (see pull-models) from the helper's --delete.
pull-output: (_rsync_with_confirm "--filter=P_models/" "--filter=P_models/**" (FLAMINGO + "/output/") "./output/")

pull-analysis:
    rsync -rtvu --progress --exclude-from=.rsyncignore {{ FLAMINGO }}/output/analysis ./output/

pull-models:
    rsync -rtvu --progress --exclude-from=.rsyncignore {{ FLAMINGO }}/output/models ./output/

pull-simulated:
    rsync -rtvu --progress --exclude-from=.rsyncignore {{ FLAMINGO }}/source_data/simulated_data ./source_data/
