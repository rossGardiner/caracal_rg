"""Review-selection policy with normal imports and deterministic classifier calls.

PyTorch is a real import dependency. No network, optimizer, or training runs.
"""

from collections import Counter

import numpy as np
import numpy.testing as npt
import pytest

from src.ActiveLearner import (
    ActiveLearner,
    ActiveLearningParameters,
    ActiveLearningStage,
    BinaryLabel,
    ReviewCandidate,
    ReviewReason,
    _ReviewSelector,
)
from src.BinaryEmbeddingClassifier import (
    BinaryClassifierParameters,
    BinaryClassifierTrainingResult,
    BinaryEmbeddingClassifier,
)
from src.CachedEmbeddingSource import CachedEmbeddingSource
from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingCorpus import EmbeddingCorpus, EmbeddingRef, EmbeddingSpaceKey


PROBABILITIES = (0.05, 0.15, 0.25, 0.39, 0.49, 0.54, 0.65, 0.75, 0.85, 0.95)
MODES = [
    pytest.param((1, 0, 0), ReviewReason.UNCERTAIN, id="uncertain-only"),
    pytest.param((0, 1, 0), ReviewReason.LIKELY_POSITIVE, id="positive-only"),
    pytest.param((0, 0, 1), ReviewReason.RANDOM, id="random-only"),
]


def parameters(fractions=(1, 0, 0), size=3, seed=7, boundary=0.5):
    return ActiveLearningParameters(
        review_queue_size=size,
        uncertainty_fraction=fractions[0],
        likely_positive_fraction=fractions[1],
        random_fraction=fractions[2],
        uncertainty_decision_boundary=boundary,
        scoring_batch_size=2,
        random_seed=seed,
    )


def scored_refs(probabilities=PROBABILITIES):
    return [(EmbeddingRef("recording-a", index), probability)
            for index, probability in enumerate(probabilities)]


def select(config, entries):
    selector = _ReviewSelector(config)
    for ref, probability in entries:
        selector.add(ref, probability)
    return selector, selector.build_queue()


class DeterministicClassifier(BinaryEmbeddingClassifier):
    """Use real classifier initialization/type checks, replacing fit/predict only."""

    def __init__(self, ratio=1):
        super().__init__(BinaryClassifierParameters(
            epochs=2, negative_to_positive_ratio=ratio, device="cpu"
        ))
        self.fit_calls = []
        self.prediction_batches = []
        self.result = None

    def fit(self, positive_vectors, negative_vectors, *, progress_callback=None):
        assert len(positive_vectors) > 0
        assert len(negative_vectors) > 0
        self.fit_calls.append((positive_vectors.copy(), negative_vectors.copy()))
        self.result = BinaryClassifierTrainingResult(
            embedding_dimension=positive_vectors.shape[1],
            positive_count=len(positive_vectors),
            available_negative_count=len(negative_vectors),
            training_negative_count=len(negative_vectors),
            final_loss=0.0,
        )
        if progress_callback is not None:
            for epoch in range(1, self.parameters.epochs + 1):
                progress_callback(epoch, self.parameters.epochs)
        return self.result

    def predict_proba(self, vectors):
        assert self.fit_calls, "Scoring must follow fit"
        assert vectors.dtype == np.float32
        assert not vectors.flags.writeable
        self.prediction_batches.append(vectors.copy())
        # The fixture's first feature is its deterministic score.
        return vectors[:, 0].copy()


@pytest.fixture
def learner_factory(tmp_path):
    created = 0

    def make(probabilities=PROBABILITIES, *, config=None, human_negative=True, ratio=1):
        nonlocal created
        created += 1
        cache = EmbeddingCache(tmp_path / f"cache-{created}")
        space = EmbeddingSpaceKey("pipeline-a", "model-a")
        entries = scored_refs(probabilities)
        positive = EmbeddingRef("human-positive", 0)
        negative = EmbeddingRef("human-negative", 0)
        entries += [(positive, 1.0)]
        if human_negative:
            entries += [(negative, 0.0)]
        for index, (ref, probability) in enumerate(entries):
            cache.save(
                recording_id=ref.recording_id, chunk_index=ref.chunk_index,
                pipeline_hash=space.pipeline_hash, embedding_name=space.embedding_name,
                values=np.array([probability, index], dtype=np.float32),
            )
        classifier = DeterministicClassifier(ratio)
        learner = ActiveLearner(
            corpus=EmbeddingCorpus(space=space, source=CachedEmbeddingSource(cache)),
            classifier=classifier, parameters=config or parameters(),
        )
        learner.label_positive(positive)
        if human_negative:
            learner.label_negative(negative)
        return learner, classifier, tuple(ref for ref, _ in scored_refs(probabilities))

    return make


def test_size_three_zero_quota_regression_through_run_round(learner_factory):
    learner, classifier, refs = learner_factory(config=parameters((1, 0, 0), size=3))

    result = learner.run_round()

    # C07 emits five: three uncertain plus one from each zero-quota bucket.
    assert len(result.review_queue) == 3
    assert tuple(candidate.ref for candidate in result.review_queue) == (refs[4], refs[5], refs[3])
    assert {candidate.reason for candidate in result.review_queue} == {ReviewReason.UNCERTAIN}
    assert result.scored_unlabelled_count == 10
    assert result.training is classifier.result
    assert learner.last_result is result


@pytest.mark.parametrize("reason", list(ReviewReason))
def test_take_with_zero_quota_does_not_append_or_mark_candidate_selected(reason):
    existing = ReviewCandidate(EmbeddingRef("existing", 0), 0.5, ReviewReason.UNCERTAIN)
    queue, selected = [existing], {existing.ref}
    extra = EmbeddingRef("extra", 0)

    _ReviewSelector._take(queue, selected, [(1.0, extra, 0.75)], 0, reason)

    assert queue == [existing]
    assert selected == {existing.ref}


@pytest.mark.parametrize("fractions, reason", MODES)
@pytest.mark.parametrize("size", [1, 2, 3, 7])
def test_zero_fraction_buckets_add_nothing(fractions, reason, size):
    selector, queue = select(parameters(fractions, size), scored_refs())
    assert selector.seen_count == 10
    assert len(queue) == size
    assert len({candidate.ref for candidate in queue}) == size
    assert {candidate.reason for candidate in queue} == {reason}


@pytest.mark.parametrize("fractions, expected", [
    ((0.6, 0.2, 0.2), {ReviewReason.UNCERTAIN: 2, ReviewReason.LIKELY_POSITIVE: 1}),
    ((0.1, 0.1, 0.8), {ReviewReason.RANDOM: 3}),
    ((0, 0.5, 0.5), {ReviewReason.LIKELY_POSITIVE: 2, ReviewReason.RANDOM: 1}),
])
def test_rounded_zero_quotas_do_not_add_extra_candidates(fractions, expected):
    _, queue = select(parameters(fractions, size=3), scored_refs())
    assert len(queue) == 3
    assert len({candidate.ref for candidate in queue}) == 3
    assert Counter(candidate.reason for candidate in queue) == expected


@pytest.mark.parametrize("fractions, reason", MODES)
def test_empty_selection_returns_empty_queue(fractions, reason):
    selector, queue = select(parameters(fractions), [])
    assert selector.seen_count == 0
    assert queue == ()


@pytest.mark.parametrize("fractions, reason", MODES)
@pytest.mark.parametrize("available", [0, 1, 2])
def test_run_round_with_fewer_unlabelled_candidates_than_requested(
    learner_factory, fractions, reason, available
):
    learner, _, refs = learner_factory(
        PROBABILITIES[:available], config=parameters(fractions, size=5)
    )
    result = learner.run_round()
    assert len(result.review_queue) == available
    assert {candidate.ref for candidate in result.review_queue} == set(refs)
    assert all(candidate.reason is reason for candidate in result.review_queue)
    assert result.scored_unlabelled_count == available


@pytest.mark.parametrize("boundary, expected_indices", [(0.5, (4, 5, 3)), (0.88, (8, 9, 7))])
def test_uncertainty_uses_configured_decision_boundary(boundary, expected_indices):
    entries = scored_refs()
    _, queue = select(parameters((1, 0, 0), boundary=boundary), entries)
    assert tuple(candidate.ref for candidate in queue) == tuple(entries[index][0] for index in expected_indices)


def test_likely_positive_bucket_ranks_highest_probabilities_first():
    entries = scored_refs()
    _, queue = select(parameters((0, 1, 0)), entries)
    assert tuple(candidate.ref for candidate in queue) == tuple(entries[index][0] for index in [9, 8, 7])
    assert [candidate.probability for candidate in queue] == [0.95, 0.85, 0.75]


def test_overlapping_uncertainty_and_positive_buckets_fill_unique_queue():
    # At boundary 1 both rankings favour the same high-probability refs.
    entries = scored_refs()
    selector, queue = select(parameters((0.5, 0.5, 0), size=6, boundary=1), entries)
    assert selector.seen_count == 10
    assert len(queue) == len({candidate.ref for candidate in queue}) == 6
    assert tuple(candidate.ref for candidate in queue) == tuple(entries[index][0] for index in [9, 8, 7, 6, 5, 4])
    assert [candidate.reason for candidate in queue] == [ReviewReason.UNCERTAIN] * 3 + [ReviewReason.LIKELY_POSITIVE] * 3


def test_random_selection_is_reproducible_and_uses_more_than_initial_reservoir():
    entries = scored_refs(np.linspace(0, 1, 50))
    _, first = select(parameters((0, 0, 1), size=3, seed=7), entries)
    _, repeated = select(parameters((0, 0, 1), size=3, seed=7), entries)
    _, different = select(parameters((0, 0, 1), size=3, seed=8), entries)
    assert first == repeated
    assert first != different
    assert len(first) == len({candidate.ref for candidate in first}) == 3
    assert any(candidate.ref.chunk_index >= 6 for candidate in first)
    probabilities = dict(entries)
    for candidate in first:
        assert candidate.probability == probabilities[candidate.ref]
        assert candidate.reason is ReviewReason.RANDOM


@pytest.mark.parametrize("fractions, reason", MODES)
def test_round_excludes_all_human_labels_and_keeps_probability_ref_pairs(
    learner_factory, fractions, reason
):
    learner, classifier, refs = learner_factory(config=parameters(fractions, size=4))
    learner.label_positive(refs[4])
    learner.label_negative(refs[9])
    labelled = learner.labelled_refs
    expected = dict(zip(refs, PROBABILITIES))

    result = learner.run_round()

    assert len(result.review_queue) == len({candidate.ref for candidate in result.review_queue}) == 4
    assert result.scored_unlabelled_count == 8
    assert not {candidate.ref for candidate in result.review_queue}.intersection(labelled)
    for candidate in result.review_queue:
        assert candidate.reason is reason
        assert candidate.probability == pytest.approx(expected[candidate.ref])
    assert learner.labelled_refs == labelled
    assert all(0 < len(batch) <= 2 for batch in classifier.prediction_batches)
    assert sum(len(batch) for batch in classifier.prediction_batches) == 12


def test_bootstrap_negatives_remain_eligible_for_review_without_becoming_human_labels(learner_factory):
    learner, classifier, refs = learner_factory(
        config=parameters((0, 0, 1), size=20), human_negative=False
    )
    labels_before = learner.labelled_refs

    result = learner.run_round()

    assert len(result.bootstrap_negative_refs) == 1
    assert set(result.bootstrap_negative_refs).issubset({candidate.ref for candidate in result.review_queue})
    assert {candidate.ref for candidate in result.review_queue} == set(refs)
    assert learner.labelled_refs == labels_before
    assert learner.negative_refs == frozenset()
    assert learner.label_for(result.bootstrap_negative_refs[0]) is None
    assert classifier.fit_calls[0][1].shape == (1, 2)


def test_round_mixes_human_and_bootstrap_negatives_without_changing_labels(learner_factory):
    learner, classifier, refs = learner_factory(ratio=3)
    labels_before = learner.labelled_refs
    result = learner.run_round()
    assert len(result.bootstrap_negative_refs) == 2
    assert set(result.bootstrap_negative_refs).issubset(refs)
    positive_vectors, negative_vectors = classifier.fit_calls[0]
    assert positive_vectors.shape == (1, 2)
    assert negative_vectors.shape == (3, 2)
    npt.assert_array_equal(negative_vectors[0], [0, 11])
    assert learner.labelled_refs == labels_before


def test_round_is_reproducible_for_fixed_seed_and_batch_size(learner_factory):
    config = parameters((0.4, 0.2, 0.4), size=5, seed=37)
    first, _, _ = learner_factory(config=config)
    second, _, _ = learner_factory(config=config)
    one, two = first.run_round(), second.run_round()
    assert one.review_queue == two.review_queue
    assert one.bootstrap_negative_refs == two.bootstrap_negative_refs
    assert one.scored_unlabelled_count == two.scored_unlabelled_count == 10


def test_round_reports_training_and_scoring_progress_without_count_scan(learner_factory, monkeypatch):
    learner, classifier, _ = learner_factory()

    def fail_count():
        raise AssertionError("Progress must not pre-count the corpus")

    monkeypatch.setattr(learner.corpus, "count", fail_count)
    progress = []
    learner.run_round(progress_callback=progress.append)
    assert [(event.completed, event.total) for event in progress
            if event.stage is ActiveLearningStage.TRAINING] == [(0, 2), (1, 2), (2, 2)]
    assert [(event.completed, event.total) for event in progress
            if event.stage is ActiveLearningStage.SCORING] == [(0, None), (12, None)]
    assert not any(event.stage is ActiveLearningStage.SAMPLING_NEGATIVES for event in progress)
    assert len(classifier.fit_calls) == 1


def test_run_round_requires_human_positive_before_training(learner_factory):
    learner, classifier, _ = learner_factory()
    learner.remove_label(next(iter(learner.positive_refs)))
    with pytest.raises(RuntimeError, match="human-positive"):
        learner.run_round()
    assert classifier.fit_calls == classifier.prediction_batches == []


def test_empty_corpus_cannot_run_a_training_round(tmp_path):
    corpus = EmbeddingCorpus(
        space=EmbeddingSpaceKey("empty-pipeline", "model-a"),
        source=CachedEmbeddingSource(EmbeddingCache(tmp_path / "empty-cache")),
    )
    classifier = DeterministicClassifier()
    learner = ActiveLearner(corpus=corpus, classifier=classifier)
    assert corpus.count() == 0
    with pytest.raises(RuntimeError, match="human-positive"):
        learner.run_round()
    assert classifier.fit_calls == classifier.prediction_batches == []


def test_round_without_available_negatives_fails_before_fit(learner_factory):
    learner, classifier, _ = learner_factory((), human_negative=False)
    with pytest.raises(RuntimeError, match="No negative embeddings"):
        learner.run_round()
    assert learner.last_result is None
    assert classifier.fit_calls == classifier.prediction_batches == []


def test_labelling_a_review_candidate_invalidates_last_result_and_excludes_it_next_round(learner_factory):
    learner, _, _ = learner_factory()
    first = learner.run_round()
    chosen = first.review_queue[0].ref
    learner.label(chosen, BinaryLabel.NEGATIVE)
    assert learner.last_result is None
    assert learner.label_for(chosen) is BinaryLabel.NEGATIVE
    second = learner.run_round()
    assert len(second.review_queue) <= learner.parameters.review_queue_size
    assert chosen not in {candidate.ref for candidate in second.review_queue}
    assert second.scored_unlabelled_count == 9
