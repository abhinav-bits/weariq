from typing import Dict, Optional

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class EmbeddingModel:
    def __init__(
        self,
        hf_model_name: str = "ai4bharat/indic-bert",
        max_length: int = 128,
        hf_token: Optional[str] = None,
    ) -> None:
        self.hf_model_name = hf_model_name
        self.max_length = max_length
        self.hf_token = hf_token
        self.tokenizer = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_model(self, model_dir: Optional[str] = None) -> None:
        model_source = model_dir if model_dir else self.hf_model_name
        auth = self.hf_token if self.hf_token else True

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_source,
            token=auth,
            use_fast=False,
        )
        self.model = AutoModel.from_pretrained(model_source, token=auth)
        self.model.to(self.device)
        self.model.eval()

    def preprocess_text(self, text: str) -> Dict[str, torch.Tensor]:
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer is not loaded. Call load_model() first.")
        encoded = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        return {key: value.to(self.device) for key, value in encoded.items()}

    def generate_embeddings(self, text: str) -> list[float]:
        if self.model is None:
            raise RuntimeError("Model is not loaded. Call load_model() first.")

        encoded = self.preprocess_text(text)
        with torch.no_grad():
            outputs = self.model(**encoded)

        token_embeddings = outputs.last_hidden_state
        attention_mask = encoded["attention_mask"].unsqueeze(-1).float()
        pooled = torch.sum(token_embeddings * attention_mask, dim=1) / torch.clamp(
            attention_mask.sum(dim=1),
            min=1e-9,
        )
        normalized = F.normalize(pooled, p=2, dim=1)
        return normalized.squeeze(0).cpu().tolist()