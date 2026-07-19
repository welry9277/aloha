import torch
from torch import nn


class FrozenDistilBertEncoder(nn.Module):
    """Frozen DistilBERT with a trainable projection to ACT hidden size."""

    def __init__(self, hidden_dim, model_name="distilbert-base-uncased"):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise ImportError(
                "Language-conditioned ACT requires transformers. Install it with "
                "`python -m pip install transformers`."
            ) from exc

        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.backbone = AutoModel.from_pretrained(model_name)
        self.backbone.requires_grad_(False)
        self.backbone.eval()
        self.projection = nn.Linear(self.backbone.config.hidden_size, hidden_dim)

    def train(self, mode=True):
        super().train(mode)
        self.backbone.eval()
        return self

    def forward(self, instructions):
        device = self.projection.weight.device
        tokens = self.tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            max_length=64,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = self.backbone(**tokens).last_hidden_state
            mask = tokens["attention_mask"].unsqueeze(-1).to(output.dtype)
            pooled = (output * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return self.projection(pooled)
