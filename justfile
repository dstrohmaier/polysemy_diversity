create_env:
    #!/usr/bin/env bash
    conda create -p ../diversity_env numpy pandas scipy scikit-learn seaborn just libstdcxx-ng -c conda-forge
    conda run -p ../diversity_env pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    conda run -p ../diversity_env pip install stanza datasets transformers tqdm evaluate accelerate

predict_efcamdat model="answerdotai/ModernBERT-large" train_dataset="wic" $CUDA_VISIBLE_DEVICES="0":
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    python apply_wic.py "output/models/{{ model }}/{{ train_dataset }}/final" source_data/efcamdat output/efcamdat/{{ train_dataset }}

push2flamingo:
    rsync -rtvu --progress --exclude-from=../../ignorelist.txt ./ ds858@flamingo.cl.cam.ac.uk:/local/scratch/ds858/wic_shift

score-wic sim_dir output_dir model="answerdotai/ModernBERT-large" $CUDA_VISIBLE_DEVICES="0":
    #!/usr/bin/env bash
    export LD_LIBRARY_PATH="../diversity_env/lib:$LD_LIBRARY_PATH"
    python score_data.py wic {{ sim_dir }} {{ output_dir }} --base-model "{{ model }}"

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
