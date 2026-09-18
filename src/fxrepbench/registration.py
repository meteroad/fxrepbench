from __future__ import annotations

from typing import Any


def build_proposal_head(
    input_dim: int,
    *,
    hidden_dim: int = 512,
    num_effects: int = 8,
    total_params: int = 22,
    max_chain_length: int = 4,
    dropout: float = 0.1,
) -> Any:
    """Build the registration MLP used by the paper baselines."""

    import torch.nn as nn

    class FxProposalHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.trunk = nn.Sequential(
                nn.LayerNorm(input_dim),
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.usage = nn.Linear(hidden_dim, num_effects)
            self.params = nn.Linear(hidden_dim, total_params)
            self.order = nn.Linear(hidden_dim, max_chain_length * (num_effects + 1))

        def forward(self, embeddings: Any) -> tuple[Any, Any, Any]:
            hidden = self.trunk(embeddings)
            order_logits = self.order(hidden).view(
                embeddings.shape[0], max_chain_length, num_effects + 1
            )
            return self.usage(hidden), self.params(hidden), order_logits

    return FxProposalHead()


def prediction_from_logits(
    usage_logits: Any,
    parameter_logits: Any,
    order_logits: Any,
) -> tuple[Any, Any, Any]:
    """Convert head logits to effect-use, parameter-center, and order probabilities."""

    import torch

    return (
        torch.sigmoid(usage_logits),
        torch.sigmoid(parameter_logits),
        torch.softmax(order_logits, dim=-1),
    )
