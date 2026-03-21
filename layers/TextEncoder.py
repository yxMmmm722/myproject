from typing import List
import hashlib
import os

import torch
import torch.nn as nn

try:
    from transformers import AutoModel, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency
    AutoModel = None
    AutoTokenizer = None


class FrozenTextEncoder(nn.Module):
    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        max_length: int = 32,
        require_transformer: bool = False,
    ):
        super().__init__()
        self.model_name = model_name
        self.max_length = max_length
        self.require_transformer = require_transformer
        self.use_transformer = False
        self.tokenizer = None
        self.model = None
        self.output_dim = 768
        self.backend = "hash"
        self.fallback_reason = ""

        if AutoModel is not None and AutoTokenizer is not None:
            try:
                use_local_files_only = os.path.isdir(model_name)
                self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=use_local_files_only)
                self.model = AutoModel.from_pretrained(model_name, local_files_only=use_local_files_only)
                self.output_dim = int(self.model.config.hidden_size)
                self.model.eval()
                for p in self.model.parameters():
                    p.requires_grad = False
                self.use_transformer = True
                self.backend = "transformer"
                print(f"[TextEncoder] Loaded transformer backend from {model_name} (dim={self.output_dim}).")
            except Exception as exc:
                # If model files are unavailable (offline/cached miss), use hash fallback.
                self.tokenizer = None
                self.model = None
                self.output_dim = 768
                self.fallback_reason = str(exc).splitlines()[0]
        else:
            self.fallback_reason = "transformers package is unavailable."

        if not self.use_transformer:
            msg = "[TextEncoder] Using hash fallback backend."
            if self.fallback_reason:
                msg += f" reason: {self.fallback_reason}"
            if self.require_transformer:
                raise RuntimeError(
                    msg + " Set --text_encoder_name to a valid local/remote HF model or unset --require_text_encoder."
                )
            print(msg)

    @torch.no_grad()
    def encode(self, texts: List[str], device, batch_size: int = 64):
        if len(texts) == 0:
            return torch.zeros((0, self.output_dim))

        if self.use_transformer:
            self.model.to(device)
            outputs = []
            for i in range(0, len(texts), batch_size):
                cur = texts[i:i + batch_size]
                tokens = self.tokenizer(
                    cur,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                )
                tokens = {k: v.to(device) for k, v in tokens.items()}
                hidden = self.model(**tokens).last_hidden_state[:, 0, :]
                outputs.append(hidden.detach().cpu())
            return torch.cat(outputs, dim=0)

        # Hash fallback for environments without transformers.
        vectors = []
        for text in texts:
            vec = torch.zeros(self.output_dim)
            for token in text.lower().split():
                idx = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16) % self.output_dim
                vec[idx] += 1.0
            vec = vec / (vec.norm(p=2) + 1e-6)
            vectors.append(vec)
        return torch.stack(vectors, dim=0)
