from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class BinaryClassifierParameters:
    """Training parameters for one binary classifier on embeddings.

    The defaults mirror the simple classifier behaviour that proved useful in
    CaracalxPerch: a linear head, Adam optimisation, class-weighted binary
    cross entropy, and a bounded number of negatives per positive example.

    This object contains only model/training configuration.  It deliberately
    contains no pipeline identity, cache paths, labels, or active-learning
    selection policy.
    """

    hidden_dimension: Optional[int] = None
    epochs: int = 20
    learning_rate: float = 1e-3
    batch_size: int = 256
    negative_to_positive_ratio: int = 5
    random_seed: int = 0
    device: str = "cpu"

    def __post_init__(self):
        hidden_dimension = self.hidden_dimension
        if hidden_dimension is not None:
            hidden_dimension = int(hidden_dimension)
            if hidden_dimension <= 0:
                raise ValueError(
                    "hidden_dimension must be greater than zero when set"
                )
            object.__setattr__(
                self,
                "hidden_dimension",
                hidden_dimension,
            )

        epochs = int(self.epochs)
        if epochs <= 0:
            raise ValueError("epochs must be greater than zero")
        object.__setattr__(self, "epochs", epochs)

        learning_rate = float(self.learning_rate)
        if learning_rate <= 0:
            raise ValueError("learning_rate must be greater than zero")
        object.__setattr__(self, "learning_rate", learning_rate)

        batch_size = int(self.batch_size)
        if batch_size <= 0:
            raise ValueError("batch_size must be greater than zero")
        object.__setattr__(self, "batch_size", batch_size)

        negative_to_positive_ratio = int(
            self.negative_to_positive_ratio
        )
        if negative_to_positive_ratio <= 0:
            raise ValueError(
                "negative_to_positive_ratio must be greater than zero"
            )
        object.__setattr__(
            self,
            "negative_to_positive_ratio",
            negative_to_positive_ratio,
        )

        object.__setattr__(
            self,
            "random_seed",
            int(self.random_seed),
        )

        device = str(self.device).strip()
        if not device:
            raise ValueError("device cannot be empty")
        object.__setattr__(self, "device", device)


@dataclass(frozen=True)
class BinaryClassifierTrainingResult:
    """Small immutable summary of one completed training run."""

    embedding_dimension: int
    positive_count: int
    available_negative_count: int
    training_negative_count: int
    final_loss: float


class _EmbeddingBinaryNetwork(nn.Module):
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


class BinaryEmbeddingClassifier:
    """Train and score a binary classifier over precomputed embeddings.

    Responsibilities:
      * own the small PyTorch classification head;
      * validate positive/negative embedding matrices;
      * cap training negatives relative to the number of positives;
      * train with class-weighted ``BCEWithLogitsLoss`` and Adam;
      * convert model logits into positive-class probabilities.

    Deliberately out of scope:
      * loading embeddings from disk;
      * deciding which examples are positive or negative;
      * sampling bootstrap negatives from a corpus;
      * choosing active-learning review candidates;
      * Qt/background-job orchestration.

    Those responsibilities belong to ``EmbeddingCorpus``, the future
    ``ActiveLearner``, and the Active Learning MVC loop respectively.
    """

    def __init__(
        self,
        parameters: Optional[BinaryClassifierParameters] = None,
    ):
        self.parameters = (
            parameters
            if parameters is not None
            else BinaryClassifierParameters()
        )

        if not isinstance(
            self.parameters,
            BinaryClassifierParameters,
        ):
            raise TypeError(
                "parameters must be BinaryClassifierParameters"
            )

        self._model = None
        self._embedding_dimension = None
        self._training_result = None

    @property
    def is_fitted(self):
        return self._model is not None

    @property
    def embedding_dimension(self):
        return self._embedding_dimension

    @property
    def training_result(self):
        return self._training_result

    def fit(
        self,
        positive_vectors,
        negative_vectors,
    ):
        """Train from positive and negative embedding matrices.

        Both inputs must be shaped ``(N, D)`` and use the same embedding
        dimension.  The caller owns example selection; this method only trains
        on the examples it receives.
        """

        positive_vectors = self._as_vector_matrix(
            positive_vectors,
            name="positive_vectors",
        )
        negative_vectors = self._as_vector_matrix(
            negative_vectors,
            name="negative_vectors",
        )

        if positive_vectors.shape[0] == 0:
            raise ValueError(
                "At least one positive training embedding is required"
            )

        if negative_vectors.shape[0] == 0:
            raise ValueError(
                "At least one negative training embedding is required"
            )

        if (
            positive_vectors.shape[1]
            != negative_vectors.shape[1]
        ):
            raise ValueError(
                "Positive and negative embeddings must have the same "
                "dimension"
            )

        embedding_dimension = positive_vectors.shape[1]
        if embedding_dimension <= 0:
            raise ValueError(
                "Embedding dimension must be greater than zero"
            )

        training_negatives = self._select_training_negatives(
            positive_count=positive_vectors.shape[0],
            negative_vectors=negative_vectors,
        )

        training_vectors = np.concatenate(
            (
                positive_vectors,
                training_negatives,
            ),
            axis=0,
        )
        training_labels = np.concatenate(
            (
                np.ones(
                    positive_vectors.shape[0],
                    dtype=np.float32,
                ),
                np.zeros(
                    training_negatives.shape[0],
                    dtype=np.float32,
                ),
            ),
            axis=0,
        )

        rng = np.random.default_rng(
            self.parameters.random_seed
        )
        permutation = rng.permutation(
            training_vectors.shape[0]
        )
        training_vectors = training_vectors[permutation]
        training_labels = training_labels[permutation]

        device = self._resolve_device()
        self._seed_torch()

        model = _EmbeddingBinaryNetwork(
            embedding_dimension=embedding_dimension,
            hidden_dimension=self.parameters.hidden_dimension,
        ).to(device)

        vector_tensor = torch.from_numpy(
            training_vectors
        ).to(
            device=device,
            dtype=torch.float32,
        )
        label_tensor = torch.from_numpy(
            training_labels
        ).to(
            device=device,
            dtype=torch.float32,
        )

        positive_count = positive_vectors.shape[0]
        training_negative_count = training_negatives.shape[0]
        positive_weight = torch.tensor(
            [training_negative_count / positive_count],
            device=device,
            dtype=torch.float32,
        )

        criterion = nn.BCEWithLogitsLoss(
            pos_weight=positive_weight
        )
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.parameters.learning_rate,
        )

        final_loss = None
        sample_count = training_vectors.shape[0]

        model.train()
        for _ in range(self.parameters.epochs):
            epoch_permutation = torch.randperm(
                sample_count,
                device=device,
            )

            epoch_loss = 0.0
            for start in range(
                0,
                sample_count,
                self.parameters.batch_size,
            ):
                indices = epoch_permutation[
                    start:start + self.parameters.batch_size
                ]
                batch_vectors = vector_tensor[indices]
                batch_labels = label_tensor[indices]

                optimizer.zero_grad()
                logits = model(batch_vectors)
                loss = criterion(logits, batch_labels)
                loss.backward()
                optimizer.step()

                epoch_loss += (
                    float(loss.detach().cpu())
                    * batch_vectors.shape[0]
                )

            final_loss = epoch_loss / sample_count

        model.eval()

        self._model = model
        self._embedding_dimension = embedding_dimension
        self._training_result = BinaryClassifierTrainingResult(
            embedding_dimension=embedding_dimension,
            positive_count=positive_count,
            available_negative_count=negative_vectors.shape[0],
            training_negative_count=training_negative_count,
            final_loss=float(final_loss),
        )

        return self._training_result

    def predict_proba(self, vectors):
        """Return positive-class probabilities for an embedding matrix."""

        if not self.is_fitted:
            raise RuntimeError(
                "BinaryEmbeddingClassifier must be fitted before prediction"
            )

        vectors = self._as_vector_matrix(
            vectors,
            name="vectors",
            allow_empty=True,
        )

        if vectors.shape[1] != self._embedding_dimension:
            raise ValueError(
                "Embedding dimension does not match the fitted classifier: "
                f"expected {self._embedding_dimension}, "
                f"received {vectors.shape[1]}"
            )

        if vectors.shape[0] == 0:
            return np.empty(
                (0,),
                dtype=np.float32,
            )

        device = self._resolve_device()
        self._model = self._model.to(device)
        self._model.eval()

        with torch.no_grad():
            vector_tensor = torch.from_numpy(
                vectors
            ).to(
                device=device,
                dtype=torch.float32,
            )
            logits = self._model(vector_tensor)
            probabilities = torch.sigmoid(logits)

        return probabilities.detach().cpu().numpy().astype(
            np.float32,
            copy=False,
        )

    def _select_training_negatives(
        self,
        *,
        positive_count,
        negative_vectors,
    ):
        max_negative_count = (
            self.parameters.negative_to_positive_ratio
            * positive_count
        )

        if negative_vectors.shape[0] <= max_negative_count:
            return negative_vectors

        rng = np.random.default_rng(
            self.parameters.random_seed
        )
        indices = rng.choice(
            negative_vectors.shape[0],
            size=max_negative_count,
            replace=False,
        )
        return np.ascontiguousarray(
            negative_vectors[indices],
            dtype=np.float32,
        )

    def _resolve_device(self):
        device = torch.device(
            self.parameters.device
        )

        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested for BinaryEmbeddingClassifier but "
                "CUDA is not available"
            )

        return device

    def _seed_torch(self):
        torch.manual_seed(
            self.parameters.random_seed
        )

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(
                self.parameters.random_seed
            )

    @staticmethod
    def _as_vector_matrix(
        vectors,
        *,
        name,
        allow_empty=False,
    ):
        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise ValueError(
                f"{name} must be a two-dimensional matrix shaped (N, D)"
            )

        if matrix.shape[1] == 0:
            raise ValueError(
                f"{name} must have a non-zero embedding dimension"
            )

        if not allow_empty and matrix.shape[0] == 0:
            raise ValueError(
                f"{name} cannot be empty"
            )

        if not np.isfinite(matrix).all():
            raise ValueError(
                f"{name} contains NaN or infinite values"
            )

        return np.ascontiguousarray(
            matrix,
            dtype=np.float32,
        )
