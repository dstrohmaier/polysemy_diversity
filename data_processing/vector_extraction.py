import logging
from dataclasses import dataclass

import numpy as np
import torch

from transformers import AutoModel, AutoTokenizer

logger = logging.getLogger("div")

# Sentences average ~35 tokens (max ~104 observed), and a corpus is at most a couple
# of hundred rows, so a whole corpus is 1-7 forward passes at this size.
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True)
class ExtractionConfig:
    hf_model_name: str


class WordVectorExtractor:
    """Extract contextual embeddings of target words from sentence contexts.

    Bundles the transformer ``model``/``tokenizer`` so they are loaded once and reused
    across extractions.
    """

    def __init__(self, model, tokenizer, device: str | torch.device = "cpu"):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device

    def get_word_vectors_from_spans(
        self,
        contexts,
        target_layers: tuple[int, ...] = (-1,),
        batch_size: int = DEFAULT_BATCH_SIZE,
    ):
        """Extract target embeddings using each context's recorded character span.

        Each context carries the authoritative span of the target occurrence
        (``ctx["start"]``/``ctx["end"]``, propagated from the WSD annotation). That
        span is mapped to subword tokens whose (optionally layer-averaged) hidden
        states are mean-pooled and L2-normalised.

        Returns an array of shape (n_found, hidden_size); rows are produced only for
        contexts whose span aligned to at least one subword.

        Row order is the input order of the surviving contexts. That is load-bearing:
        the scorers' ``equalise_indices`` down-samples *positionally* from a seeded
        RNG, so reordering (or dropping a different number of) rows would silently
        change which occurrences get scored. Contexts are therefore batched in
        contiguous slices and never sorted -- in particular not bucketed by length,
        which is the tempting optimisation that would break this.
        """
        self.model.eval()

        # Partition first, carrying nothing but the order: batching must not change
        # which contexts survive, only how many forward passes embed them.
        kept: list[tuple[str, int, int]] = []
        n_skipped_no_span = 0
        for ctx in contexts:
            offset_start = ctx.get("start")
            offset_end = ctx.get("end")
            if offset_start is None or offset_end is None or offset_end <= offset_start:
                n_skipped_no_span += 1
                continue
            kept.append((ctx["sentence"], int(offset_start), int(offset_end)))

        vectors = []
        n_skipped_unaligned = 0
        for start in range(0, len(kept), batch_size):
            chunk = kept[start : start + batch_size]
            embedded = self._embed_batch(
                [text for text, _, _ in chunk],
                [(s, e) for _, s, e in chunk],
                target_layers,
            )
            for word_vector in embedded:
                if word_vector is None:
                    n_skipped_unaligned += 1
                    continue
                vectors.append(word_vector)

        n_skipped = n_skipped_no_span + n_skipped_unaligned
        if n_skipped:
            logger.info(
                "get_word_vectors_from_spans: extracted %d/%d, skipped %d "
                "(%d missing span, %d unaligned to subwords)",
                len(vectors),
                len(contexts),
                n_skipped,
                n_skipped_no_span,
                n_skipped_unaligned,
            )

        if vectors:
            return np.array(vectors)
        return np.empty((0, self.model.config.hidden_size))

    def _embed_batch(
        self,
        texts: list[str],
        spans: list[tuple[int, int]],
        target_layers: tuple[int, ...],
    ) -> list[np.ndarray | None]:
        """Embed one batch of ``(text, span)`` pairs in a single forward pass.

        Returns one entry per input, positionally aligned with ``texts``: the
        mean-pooled, L2-normalised hidden state of the subwords overlapping the span,
        or ``None`` where the span aligned to no subword (e.g. it fell past the
        truncation length). Callers rely on that alignment to keep row order, so the
        length of the result is asserted rather than assumed.
        """
        assert len(texts) == len(spans)
        if not texts:
            return []

        encoding = self.tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024,
            return_offsets_mapping=True,
        )

        offsets = encoding.pop("offset_mapping").tolist()  # (B, L, 2)
        attention_mask = encoding["attention_mask"].tolist()  # (B, L)
        model_inputs = {k: v.to(self.device) for k, v in encoding.items()}

        with torch.no_grad():
            outputs = self.model(**model_inputs, output_hidden_states=True)

        # outputs.hidden_states is a tuple (embeddings, layer_1, ..., layer_N), each
        # (B, L, H). Average the requested layers (default: just the last), keeping
        # the batch axis -- indexing a single row here is what limited this to one
        # sentence per forward pass.
        hidden_states = torch.mean(
            torch.stack(
                [outputs.hidden_states[layer] for layer in target_layers],
                dim=0,
            ),
            dim=0,
        )  # (B, L, H)

        results: list[np.ndarray | None] = []
        for row, (offset_start, offset_end) in enumerate(spans):
            word_token_indices = [
                i
                for i, (token_start, token_end) in enumerate(offsets[row])
                # A padded position carries offsets (0, 0) and so is already excluded
                # by the token_end > token_start guard; the mask makes that explicit,
                # since pooling a pad would silently corrupt the shorter rows of a
                # mixed-length batch.
                if attention_mask[row][i]
                and token_end > token_start  # skip special tokens (span (0, 0))
                and not (token_end <= offset_start or token_start >= offset_end)
            ]
            if not word_token_indices:
                results.append(None)
                continue

            word_vector = (
                hidden_states[row][word_token_indices].mean(dim=0).detach().cpu().numpy()
            )
            norm = np.linalg.norm(word_vector)
            if norm > 0:
                word_vector = word_vector / norm
            results.append(word_vector)

        assert len(results) == len(texts), "batch must return one entry per input"
        return results

    @classmethod
    def from_config(cls, config: ExtractionConfig):
        model = AutoModel.from_pretrained(config.hf_model_name)
        tokenizer = AutoTokenizer.from_pretrained(config.hf_model_name)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        return cls(model, tokenizer, device)
