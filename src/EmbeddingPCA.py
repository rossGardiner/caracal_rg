from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EmbeddingPCAResult:
    """
    Display-ready 2D PCA result.

    coordinates contains one (PC1, PC2) row for every embedding supplied
    to calculate_embedding_pca(). explained_variance contains the fraction
    of variance explained by PC1 and PC2 respectively.
    """

    coordinates: np.ndarray
    explained_variance: np.ndarray


def calculate_embedding_pca(
    vectors,
):
    """
    Calculate a two-dimensional PCA projection using NumPy SVD.

    This function contains no Qt or PyQtGraph code so it can run safely in
    a background worker. The caller supplies a snapshot/sequence of
    one-dimensional embedding vectors.
    """

    vectors = tuple(
        vectors
    )

    if not vectors:
        raise ValueError(
            "At least one embedding vector is required"
        )

    matrix = np.stack(
        [
            np.asarray(
                vector,
                dtype=np.float64,
            ).reshape(-1)
            for vector in vectors
        ],
        axis=0,
    )

    sample_count = matrix.shape[0]

    if sample_count == 1:
        return EmbeddingPCAResult(
            coordinates=np.zeros(
                (1, 2),
                dtype=np.float64,
            ),
            explained_variance=np.zeros(
                2,
                dtype=np.float64,
            ),
        )

    centred = (
        matrix
        - matrix.mean(
            axis=0,
            keepdims=True,
        )
    )

    (
        u,
        singular_values,
        _,
    ) = np.linalg.svd(
        centred,
        full_matrices=False,
    )

    component_count = min(
        2,
        len(
            singular_values
        ),
    )

    coordinates = (
        u[
            :,
            :component_count
        ]
        * singular_values[
            :component_count
        ]
    )

    if component_count < 2:
        coordinates = np.pad(
            coordinates,
            (
                (0, 0),
                (
                    0,
                    2 - component_count,
                ),
            ),
        )

    variance = (
        singular_values ** 2
    )

    total_variance = (
        variance.sum()
    )

    if total_variance > 0:
        explained_variance = (
            variance
            / total_variance
        )
    else:
        explained_variance = np.zeros_like(
            variance
        )

    if len(explained_variance) < 2:
        explained_variance = np.pad(
            explained_variance,
            (
                0,
                2 - len(
                    explained_variance
                ),
            ),
        )

    return EmbeddingPCAResult(
        coordinates=coordinates,
        explained_variance=(
            explained_variance[:2]
        ),
    )
