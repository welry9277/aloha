"""Deprecated standalone prototype.

Training now uses the vendored official ACT implementation under
``third_party/official_act``. This module remains only as a reference and is
not imported by ``train_language_act.py``.
"""

import math

import torch
from torch import nn

from task_instructions import tokenize_instruction


def build_vocabulary(instructions):
    tokens = sorted({token for text in instructions for token in tokenize_instruction(text)})
    return {"<pad>": 0, "<unk>": 1, **{token: i + 2 for i, token in enumerate(tokens)}}


class WordGRUEncoder(nn.Module):
    def __init__(self, vocabulary, hidden_dim):
        super().__init__()
        self.vocabulary = dict(vocabulary)
        self.embedding = nn.Embedding(len(vocabulary), hidden_dim, padding_idx=0)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

    def _tokenize_batch(self, instructions, device):
        sequences = []
        for instruction in instructions:
            ids = [
                self.vocabulary.get(token, self.vocabulary["<unk>"])
                for token in tokenize_instruction(instruction)
            ]
            sequences.append(ids or [self.vocabulary["<unk>"]])
        max_length = max(len(sequence) for sequence in sequences)
        tokens = torch.zeros(len(sequences), max_length, dtype=torch.long, device=device)
        lengths = torch.tensor([len(sequence) for sequence in sequences], device=device)
        for index, sequence in enumerate(sequences):
            tokens[index, : len(sequence)] = torch.tensor(sequence, device=device)
        return tokens, lengths

    def forward(self, instructions):
        device = self.embedding.weight.device
        tokens, lengths = self._tokenize_batch(instructions, device)
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(
            embedded, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, hidden = self.gru(packed)
        return hidden[-1]


class FrozenDistilBertEncoder(nn.Module):
    def __init__(self, hidden_dim, model_name="distilbert-base-uncased"):
        super().__init__()
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "DistilBERT mode requires transformers. Install it with: "
                "python -m pip install transformers"
            ) from error
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.encoder.requires_grad_(False)
        self.encoder.eval()
        self.projection = nn.Linear(self.encoder.config.hidden_size, hidden_dim)

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        return self

    def forward(self, instructions):
        device = self.projection.weight.device
        tokens = self.tokenizer(
            list(instructions),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)
        with torch.no_grad():
            output = self.encoder(**tokens).last_hidden_state[:, 0]
        return self.projection(output)


class SharedCameraEncoder(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 32, 5, stride=2, padding=2),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.GELU(),
            nn.Conv2d(128, hidden_dim, 3, stride=2, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, images):
        batch, cameras, channels, height, width = images.shape
        features = self.backbone(images.reshape(batch * cameras, channels, height, width))
        return features.flatten(1).reshape(batch, cameras, -1)


class LanguageConditionedACT(nn.Module):
    def __init__(
        self,
        chunk_size=10,
        action_dim=14,
        state_dim=14,
        hidden_dim=256,
        nheads=8,
        encoder_layers=4,
        decoder_layers=4,
        dim_feedforward=1024,
        text_encoder="gru",
        vocabulary=None,
        text_model_name="distilbert-base-uncased",
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.nheads = nheads
        self.encoder_layers = encoder_layers
        self.decoder_layers = decoder_layers
        self.dim_feedforward = dim_feedforward
        self.text_encoder_name = text_encoder
        self.vocabulary = vocabulary
        self.text_model_name = text_model_name

        self.camera_encoder = SharedCameraEncoder(hidden_dim)
        self.state_projection = nn.Linear(state_dim, hidden_dim)
        self.latent_projection = nn.Linear(hidden_dim, hidden_dim)

        if text_encoder == "gru":
            if not vocabulary:
                raise ValueError("GRU text encoder requires a vocabulary")
            self.text_encoder = WordGRUEncoder(vocabulary, hidden_dim)
        elif text_encoder == "distilbert":
            self.text_encoder = FrozenDistilBertEncoder(hidden_dim, text_model_name)
        else:
            raise ValueError(f"Unknown text encoder: {text_encoder}")

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.policy_encoder = nn.TransformerEncoder(encoder_layer, encoder_layers)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.action_decoder = nn.TransformerDecoder(decoder_layer, decoder_layers)
        self.memory_positions = nn.Parameter(torch.randn(1, 6, hidden_dim) * 0.02)
        self.action_queries = nn.Parameter(torch.randn(1, chunk_size, hidden_dim) * 0.02)
        self.action_head = nn.Linear(hidden_dim, action_dim)

        self.action_embedding = nn.Linear(action_dim, hidden_dim)
        self.posterior_state_embedding = nn.Linear(state_dim, hidden_dim)
        posterior_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=nheads,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.posterior_encoder = nn.TransformerEncoder(posterior_layer, 2)
        self.posterior_cls = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        self.posterior_positions = nn.Parameter(
            torch.randn(1, chunk_size + 2, hidden_dim) * 0.02
        )
        self.latent_stats = nn.Linear(hidden_dim, hidden_dim * 2)

    def config_dict(self):
        return {
            "chunk_size": self.chunk_size,
            "action_dim": self.action_dim,
            "state_dim": 14,
            "hidden_dim": self.hidden_dim,
            "nheads": self.nheads,
            "encoder_layers": self.encoder_layers,
            "decoder_layers": self.decoder_layers,
            "dim_feedforward": self.dim_feedforward,
            "text_encoder": self.text_encoder_name,
            "vocabulary": self.vocabulary,
            "text_model_name": self.text_model_name,
        }

    def _posterior(self, state, actions, is_pad):
        batch = state.shape[0]
        cls = self.posterior_cls.expand(batch, -1, -1)
        state_token = self.posterior_state_embedding(state).unsqueeze(1)
        action_tokens = self.action_embedding(actions)
        tokens = torch.cat([cls, state_token, action_tokens], dim=1)
        tokens = tokens + self.posterior_positions[:, : tokens.shape[1]]
        prefix_pad = torch.zeros(batch, 2, dtype=torch.bool, device=state.device)
        padding_mask = torch.cat([prefix_pad, is_pad], dim=1)
        encoded = self.posterior_encoder(tokens, src_key_padding_mask=padding_mask)
        mean, log_variance = self.latent_stats(encoded[:, 0]).chunk(2, dim=-1)
        standard_deviation = torch.exp(0.5 * log_variance)
        latent = mean + standard_deviation * torch.randn_like(standard_deviation)
        return latent, mean, log_variance

    def forward(self, images, state, instructions, actions=None, is_pad=None):
        if actions is not None:
            latent, mean, log_variance = self._posterior(state, actions, is_pad)
        else:
            latent = torch.zeros(state.shape[0], self.hidden_dim, device=state.device)
            mean = log_variance = None

        camera_tokens = self.camera_encoder(images)
        state_token = self.state_projection(state).unsqueeze(1)
        language_token = self.text_encoder(instructions).unsqueeze(1)
        latent_token = self.latent_projection(latent).unsqueeze(1)
        memory = torch.cat(
            [camera_tokens, state_token, language_token, latent_token], dim=1
        )
        memory = self.policy_encoder(memory + self.memory_positions)
        queries = self.action_queries.expand(state.shape[0], -1, -1)
        decoded = self.action_decoder(queries, memory)
        return self.action_head(decoded), mean, log_variance


def act_loss(predicted_actions, target_actions, is_pad, mean, log_variance, kl_weight):
    valid = (~is_pad).unsqueeze(-1).to(predicted_actions.dtype)
    l1 = (torch.abs(predicted_actions - target_actions) * valid).sum()
    l1 = l1 / (valid.sum().clamp_min(1.0) * predicted_actions.shape[-1])
    if mean is None:
        kl = predicted_actions.new_zeros(())
    else:
        kl = -0.5 * (1.0 + log_variance - mean.square() - log_variance.exp())
        kl = kl.mean()
    return l1 + kl_weight * kl, l1, kl
