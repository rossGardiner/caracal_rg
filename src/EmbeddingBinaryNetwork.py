"""PyTorch model head; imported only when fitting/restoring a classifier."""

from typing import Optional

from torch import nn


class EmbeddingBinaryNetwork(nn.Module):
    """Tiny PyTorch head placed on top of fixed embedding vectors."""

    def __init__(
        self,
        embedding_dimension: int,
        hidden_dimension: Optional[int],
    ):
        super().__init__()

        if hidden_dimension is None:
            self.network = nn.Linear(
                embedding_dimension,
                1,
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(
                    embedding_dimension,
                    hidden_dimension,
                ),
                nn.ReLU(),
                nn.Linear(
                    hidden_dimension,
                    1,
                ),
            )

    def forward(self, vectors):
        return self.network(vectors).squeeze(-1)
