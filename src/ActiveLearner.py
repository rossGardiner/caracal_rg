from dataclasses import dataclass
from enum import Enum
import heapq
import random
from typing import Optional

import numpy as np

from src.BinaryEmbeddingClassifier import (
    BinaryClassifierTrainingResult,
    BinaryEmbeddingClassifier,
)
from src.EmbeddingCorpus import EmbeddingCorpus, EmbeddingRef


class BinaryLabel(Enum):
    """Human label assigned to one embedding."""

    POSITIVE = 1
    NEGATIVE = 0


class ReviewReason(Enum):
    """Reason an unlabeled embedding was selected for review."""

    UNCERTAIN = "uncertain"
    LIKELY_POSITIVE = "likely_positive"
    RANDOM = "random"


@dataclass(frozen=True)
class ActiveLearningParameters:
    """Configuration for corpus scoring and review selection."""

    review_queue_size: int = 100
    uncertainty_fraction: float = 0.6
    likely_positive_fraction: float = 0.2
    random_fraction: float = 0.2
    uncertainty_decision_boundary: float = 0.5
    scoring_batch_size: int = 8192
    random_seed: int = 0

    def __post_init__(self):
        queue_size = int(self.review_queue_size)
        batch_size = int(self.scoring_batch_size)
        fractions = (
            float(self.uncertainty_fraction),
            float(self.likely_positive_fraction),
            float(self.random_fraction),
        )
        boundary = float(self.uncertainty_decision_boundary)

        if queue_size <= 0:
            raise ValueError("review_queue_size must be greater than zero")
        if batch_size <= 0:
            raise ValueError("scoring_batch_size must be greater than zero")
        if any(value < 0.0 for value in fractions):
            raise ValueError("review fractions cannot be negative")
        if sum(fractions) <= 0.0:
            raise ValueError("at least one review fraction must be positive")
        if not 0.0 <= boundary <= 1.0:
            raise ValueError(
                "uncertainty_decision_boundary must be between 0 and 1"
            )

        object.__setattr__(self, "review_queue_size", queue_size)
        object.__setattr__(self, "scoring_batch_size", batch_size)
        object.__setattr__(self, "uncertainty_fraction", fractions[0])
        object.__setattr__(self, "likely_positive_fraction", fractions[1])
        object.__setattr__(self, "random_fraction", fractions[2])
        object.__setattr__(self, "uncertainty_decision_boundary", boundary)
        object.__setattr__(self, "random_seed", int(self.random_seed))


@dataclass(frozen=True)
class ReviewCandidate:
    """One unlabeled example selected for human review."""

    ref: EmbeddingRef
    probability: float
    reason: ReviewReason

    def __post_init__(self):
        if not isinstance(self.ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef")
        if not isinstance(self.reason, ReviewReason):
            raise TypeError("reason must be a ReviewReason")

        probability = float(self.probability)
        if not 0.0 <= probability <= 1.0:
            raise ValueError("probability must be between 0 and 1")
        object.__setattr__(self, "probability", probability)


@dataclass(frozen=True)
class ActiveLearningRoundResult:
    """Result of one train, score, and candidate-selection round."""

    training: BinaryClassifierTrainingResult
    bootstrap_negative_refs: tuple[EmbeddingRef, ...]
    review_queue: tuple[ReviewCandidate, ...]
    scored_unlabelled_count: int


class ActiveLearner:
    """Run active learning for one binary detector in one embedding space.

    The learner owns human labels and the active-learning policy.  It uses an
    ``EmbeddingCorpus`` for bounded disk access and a
    ``BinaryEmbeddingClassifier`` for model training/prediction.

    Bootstrap negatives are temporary training assumptions, never human
    labels.  They remain eligible for review in later rounds.
    """

    def __init__(
        self,
        *,
        corpus: EmbeddingCorpus,
        classifier: BinaryEmbeddingClassifier,
        parameters: Optional[ActiveLearningParameters] = None,
    ):
        if not isinstance(corpus, EmbeddingCorpus):
            raise TypeError("corpus must be an EmbeddingCorpus")
        if not isinstance(classifier, BinaryEmbeddingClassifier):
            raise TypeError(
                "classifier must be a BinaryEmbeddingClassifier"
            )

        self.corpus = corpus
        self.classifier = classifier
        self.parameters = parameters or ActiveLearningParameters()
        if not isinstance(self.parameters, ActiveLearningParameters):
            raise TypeError(
                "parameters must be ActiveLearningParameters"
            )

        self._labels = {}
        self._last_result = None

    @property
    def embedding_space(self):
        return self.corpus.space

    @property
    def positive_refs(self):
        return self._refs_with_label(BinaryLabel.POSITIVE)

    @property
    def negative_refs(self):
        return self._refs_with_label(BinaryLabel.NEGATIVE)

    @property
    def labelled_refs(self):
        return frozenset(self._labels)

    @property
    def last_result(self):
        return self._last_result

    def label_for(self, ref: EmbeddingRef):
        """Return the human label for one ref, or ``None`` if unlabelled."""

        if not isinstance(ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef")
        return self._labels.get(ref)

    def label(self, ref: EmbeddingRef, label: BinaryLabel):
        """Set or replace the human label for one embedding."""

        if not isinstance(ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef")
        if not isinstance(label, BinaryLabel):
            raise TypeError("label must be a BinaryLabel")

        self._labels[ref] = label
        self._last_result = None

    def label_positive(self, ref: EmbeddingRef):
        self.label(ref, BinaryLabel.POSITIVE)

    def label_negative(self, ref: EmbeddingRef):
        self.label(ref, BinaryLabel.NEGATIVE)

    def remove_label(self, ref: EmbeddingRef):
        if not isinstance(ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef")
        self._labels.pop(ref, None)
        self._last_result = None

    def run_round(self):
        """Train from current labels and produce a fresh review queue."""

        positive_refs = tuple(sorted(self.positive_refs))
        if not positive_refs:
            raise RuntimeError(
                "At least one human-positive embedding is required before "
                "active learning can run"
            )

        negative_refs, bootstrap_batch = self._training_negatives(
            positive_count=len(positive_refs)
        )
        positive_batch = self.corpus.load_refs(positive_refs)
        human_negative_batch = self.corpus.load_refs(negative_refs)
        negative_vectors = self._join_negative_vectors(
            human_negative_batch.vectors,
            bootstrap_batch.vectors,
        )

        if negative_vectors.shape[0] == 0:
            raise RuntimeError(
                "No negative embeddings are available for classifier training"
            )

        training = self.classifier.fit(
            positive_batch.vectors,
            negative_vectors,
        )
        review_queue, scored_count = self._score_corpus()

        self._last_result = ActiveLearningRoundResult(
            training=training,
            bootstrap_negative_refs=bootstrap_batch.refs,
            review_queue=review_queue,
            scored_unlabelled_count=scored_count,
        )
        return self._last_result

    def _training_negatives(self, positive_count):
        desired_count = (
            positive_count
            * self.classifier.parameters.negative_to_positive_ratio
        )
        human_refs = sorted(self.negative_refs)

        if len(human_refs) > desired_count:
            rng = random.Random(self.parameters.random_seed)
            human_refs = sorted(rng.sample(human_refs, desired_count))

        bootstrap_count = max(0, desired_count - len(human_refs))
        bootstrap_batch = self.corpus.sample(
            bootstrap_count,
            seed=self.parameters.random_seed,
            exclude_refs=self.labelled_refs,
        )
        return tuple(human_refs), bootstrap_batch

    def _score_corpus(self):
        selector = _ReviewSelector(self.parameters)
        labelled = self.labelled_refs

        for batch in self.corpus.iter_batches(
            batch_size=self.parameters.scoring_batch_size
        ):
            probabilities = self.classifier.predict_proba(batch.vectors)
            for ref, probability in zip(batch.refs, probabilities):
                if ref not in labelled:
                    selector.add(ref, float(probability))

        return selector.build_queue(), selector.seen_count

    def _refs_with_label(self, target):
        return frozenset(
            ref
            for ref, label in self._labels.items()
            if label is target
        )

    @staticmethod
    def _join_negative_vectors(human_vectors, bootstrap_vectors):
        if human_vectors.shape[0] == 0:
            return bootstrap_vectors
        if bootstrap_vectors.shape[0] == 0:
            return human_vectors
        if human_vectors.shape[1] != bootstrap_vectors.shape[1]:
            raise ValueError(
                "Human and bootstrap negative embeddings have different "
                "dimensions"
            )
        return np.concatenate(
            (human_vectors, bootstrap_vectors),
            axis=0,
        )


class _BoundedBest:
    """Keep only the best K scored candidates from a stream."""

    def __init__(self, size):
        self.size = int(size)
        self._heap = []

    def add(self, quality, ref, probability):
        entry = (float(quality), ref, float(probability))
        if len(self._heap) < self.size:
            heapq.heappush(self._heap, entry)
        elif entry > self._heap[0]:
            heapq.heapreplace(self._heap, entry)

    def best_first(self):
        return sorted(self._heap, reverse=True)


class _ReviewSelector:
    """Bounded-memory uncertainty/positive/random candidate selector."""

    def __init__(self, parameters: ActiveLearningParameters):
        self.parameters = parameters
        reserve_size = parameters.review_queue_size * 2
        self._uncertain = _BoundedBest(reserve_size)
        self._positive = _BoundedBest(reserve_size)
        self._random = []
        self._random_size = reserve_size
        self._rng = random.Random(parameters.random_seed)
        self.seen_count = 0

    def add(self, ref: EmbeddingRef, probability: float):
        self.seen_count += 1
        uncertainty = abs(
            probability - self.parameters.uncertainty_decision_boundary
        )
        self._uncertain.add(-uncertainty, ref, probability)
        self._positive.add(probability, ref, probability)
        self._add_random(ref, probability)

    def build_queue(self):
        target_size = min(
            self.parameters.review_queue_size,
            self.seen_count,
        )
        counts = self._bucket_counts(target_size)
        selected = set()
        queue = []

        self._take(
            queue,
            selected,
            self._uncertain.best_first(),
            counts[ReviewReason.UNCERTAIN],
            ReviewReason.UNCERTAIN,
        )
        self._take(
            queue,
            selected,
            self._positive.best_first(),
            counts[ReviewReason.LIKELY_POSITIVE],
            ReviewReason.LIKELY_POSITIVE,
        )

        self._rng.shuffle(self._random)
        random_candidates = [
            (0.0, ref, probability)
            for ref, probability in self._random
        ]
        self._take(
            queue,
            selected,
            random_candidates,
            counts[ReviewReason.RANDOM],
            ReviewReason.RANDOM,
        )

        # Bucket overlap can cause a shortfall.  The uncertainty reserve keeps
        # extra candidates specifically so we can fill without rescanning disk.
        if len(queue) < target_size:
            self._take(
                queue,
                selected,
                self._uncertain.best_first(),
                target_size - len(queue),
                ReviewReason.UNCERTAIN,
            )

        return tuple(queue)

    def _add_random(self, ref, probability):
        item = (ref, probability)
        if len(self._random) < self._random_size:
            self._random.append(item)
            return

        replacement = self._rng.randrange(self.seen_count)
        if replacement < self._random_size:
            self._random[replacement] = item

    def _bucket_counts(self, total):
        fractions = {
            ReviewReason.UNCERTAIN: self.parameters.uncertainty_fraction,
            ReviewReason.LIKELY_POSITIVE: (
                self.parameters.likely_positive_fraction
            ),
            ReviewReason.RANDOM: self.parameters.random_fraction,
        }
        fraction_total = sum(fractions.values())
        raw = {
            reason: total * fraction / fraction_total
            for reason, fraction in fractions.items()
        }
        counts = {reason: int(value) for reason, value in raw.items()}
        remainder = total - sum(counts.values())
        order = sorted(
            raw,
            key=lambda reason: raw[reason] - counts[reason],
            reverse=True,
        )
        for reason in order[:remainder]:
            counts[reason] += 1
        return counts

    @staticmethod
    def _take(queue, selected, candidates, count, reason):
        added = 0
        for _, ref, probability in candidates:
            if ref in selected:
                continue
            queue.append(
                ReviewCandidate(
                    ref=ref,
                    probability=probability,
                    reason=reason,
                )
            )
            selected.add(ref)
            added += 1
            if added >= count:
                break
