"""Bidirectional, pre-normalized policy/value encoder."""
from dataclasses import dataclass, asdict
import torch
from torch import nn

@dataclass
class Config:
    width: int = 384
    layers: int = 6
    heads: int = 6
    dropout: float = 0.0

CONFIGS = {
    "small": Config(384, 6, 6),
    "medium": Config(640, 10, 10),
    "large": Config(896, 16, 14),
}

class PolicyValue(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token = nn.Embedding(25, config.width)
        self.position = nn.Parameter(torch.randn(69, config.width) * .02)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(config.width, config.heads,
                dim_feedforward=4*config.width, dropout=config.dropout,
                activation="gelu", batch_first=True, norm_first=True)
            for _ in range(config.layers)])
        self.norm = nn.LayerNorm(config.width)
        self.policy = nn.Linear(config.width, 1968)
        self.value = nn.Linear(config.width, 1)

    def forward(self, tokens):
        x = self.token(tokens.long()) + self.position
        for block in self.blocks:
            x = block(x)
        x = self.norm(x).mean(dim=1)
        return self.policy(x), self.value(x).squeeze(-1)
