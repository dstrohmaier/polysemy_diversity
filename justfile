unit-test:
    python -m unittest discover -s './tests' -p '*_tests.py'

create-env:
    #!/usr/bin/env bash
    conda create -p ../diversity_env just libstdcxx-ng -c conda-forge
    conda run -p ../diversity_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    conda run -p ../diversity_env pip install jax[cuda13] numpy pandas scipy scikit-learn seaborn datasets transformers tqdm evaluate accelerate nltk

# ---- Vocab Creation

create-vocab vocab="most_diverse" n="100":
    #!/usr/bin/env bash
    python create_vocabs.py source_data/vocabs {{ vocab }} -n {{ n }}

# --- Simulation

simulate-target-verbs:
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    python simulate_data.py source_data/word_sense_disambigation_corpora source_data/vocabs/target_verbs.json source_data/simulated_data/target_verbs

simulate-most-diverse pos:
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    python simulate_data.py source_data/word_sense_disambigation_corpora source_data/vocabs/most_diverse_{{ pos }}.json source_data/simulated_data/most_diverse_{{ pos }}

simulate-most-diverse-all:
    #!/usr/bin/env bash
    for pos in noun verb adj adv; do
        just simulate-most-diverse "$pos"
    done

# --- Training WiC

train model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES="0":
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    python create_wic_model.py "{{ model }}" source_data output/models

train-tempowic model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES="0":
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    python create_wic_model.py "{{ model }}" source_data output/models --dataset tempowic

train-wic-tempowic model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES="0":
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    python create_wic_model.py "{{ model }}" source_data output/models --dataset wic+tempowic

predict_efcamdat model="answerdotai/ModernBERT-large" train_dataset="wic" $CUDA_VISIBLE_DEVICES="0":
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    # Training stores the model name slash-replaced (model_name.replace("/", "--")).
    model_path="{{ model }}"
    model_path="${model_path//\//--}"
    python apply_wic.py "output/models/${model_path}/{{ train_dataset }}/final" source_data/efcamdat output/efcamdat/{{ train_dataset }}

# --- Scoring

score-wic sim_dir output_dir model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES="0":
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    python score_data.py wic {{ sim_dir }} {{ output_dir }} --base-model "{{ model }}"

score-wic-all:
    #!/usr/bin/env bash
    for sim_dir in source_data/simulated_data/*/; do
        name=$(basename "$sim_dir")
        just score-wic "$sim_dir" "output/scores/$name"
    done

score-vmf sim_dir output_dir model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES="0":
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    python score_data.py vmf {{ sim_dir }} {{ output_dir }} --hf-model-name "{{ model }}"

score-vmf-all:
    #!/usr/bin/env bash
    for sim_dir in source_data/simulated_data/*/; do
        name=$(basename "$sim_dir")
        just score-vmf "$sim_dir" "output/scores/$name"
    done

# --- Data Transfer

push2flamingo:
    rsync -rtvu --progress --exclude-from=../../ignorelist.txt ./ ds858@flamingo.cl.cam.ac.uk:/local/scratch/ds858/wic_shift
