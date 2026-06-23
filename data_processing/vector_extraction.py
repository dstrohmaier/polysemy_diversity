from dataclasses import dataclass

import numpy as np
import torch

import spacy
from transformers import AutoModel, AutoTokenizer


@dataclass(frozen=True)
class ExtractionConfig:
    hf_model_name: str
    spacy_model_name: str = "en_core_web_sm"


class WordVectorExtractor:
    """Extract contextual embeddings of target words from sentence contexts.

    Bundles the transformer ``model``/``tokenizer`` and the spaCy ``nlp`` pipeline
    so they are loaded once and reused across extractions.
    """

    def __init__(self, model, tokenizer, nlp, device: str | torch.device = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.nlp = nlp
        self.device = device

    def get_word_vectors(
        self,
        contexts,
        word,
        target_pos,
        target_layers: tuple[int, ...] = (-1,),
    ):
        """Extract embeddings of ``word`` (with ``target_pos``) from each context.

        For each context, the target token is located by matching ``word``/
        ``target_pos`` with spaCy, its character span is mapped to subword tokens,
        and the (optionally layer-averaged) hidden states of those subwords are
        mean-pooled and L2-normalised.

        Returns an array of shape (n_found, hidden_size); rows are produced only
        for contexts where the target word was found and aligned to a subword.
        """

        self.model.eval()
        vectors = []
        for ctx in contexts:
            text = ctx["sentence"]
            doc = self.nlp(text)

            spacy_token = next(
                (
                    t
                    for t in doc
                    if t.lemma_.lower() == word.lower() and t.pos_ == target_pos
                ),
                None,
            )
            if spacy_token is None:
                continue

            offset_start = spacy_token.idx
            offset_end = spacy_token.idx + len(spacy_token.text)

            encoding = self.tokenizer(
                text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
                return_offsets_mapping=True,
            )

            offsets = encoding.pop("offset_mapping")[0].tolist()
            model_inputs = {k: v.to(self.device) for k, v in encoding.items()}

            with torch.no_grad():
                outputs = self.model(**model_inputs, output_hidden_states=True)

            # outputs.hidden_states is a tuple (embeddings, layer_1, ..., layer_N).
            # Average the requested layers (default: just the last).
            hidden_states = torch.mean(
                torch.stack(
                    [outputs.hidden_states[layer][0] for layer in target_layers],
                    dim=0,
                ),
                dim=0,
            )

            word_token_indices = [
                i
                for i, (token_start, token_end) in enumerate(offsets)
                if token_end > token_start  # skip special tokens (span (0, 0))
                and not (token_end <= offset_start or token_start >= offset_end)
            ]
            if not word_token_indices:
                continue

            word_vector = (
                hidden_states[word_token_indices].mean(dim=0).detach().cpu().numpy()
            )
            norm = np.linalg.norm(word_vector)
            if norm > 0:
                word_vector = word_vector / norm
            vectors.append(word_vector)

        return np.array(vectors) if vectors else np.array([]).reshape(0, -1)

    @classmethod
    def from_config(cls, config: ExtractionConfig):
        model = AutoModel.from_pretrained(config.hf_model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)

        nlp = spacy.load(config.spacy_model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(model, tokenizer, nlp, device)
