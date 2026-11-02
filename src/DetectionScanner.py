"""Explicit, bounded-memory detection scanning of an already fitted classifier."""

from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np

from src.BinaryEmbeddingClassifier import BinaryEmbeddingClassifier
from src.Detection import Detection, DetectionRun
from src.DetectionStore import DetectionStore
from src.EmbeddingCorpus import EmbeddingCorpus


@dataclass(frozen=True)
class DetectionScanProgress:
    scored_count: int
    detection_count: int


class DetectionScanCancelled(RuntimeError):
    """Raised when a user cancels a scan; its SQL transaction rolls back."""


def fingerprint_classifier_state(parameters, state):
    """Fingerprint saved weights without instantiating or fitting a network.

    The hashing algorithm intentionally matches classifier_fingerprint from
    Patches 1-3, so existing detection runs need no database migration.
    """
    if state is None:
        raise RuntimeError("Cannot fingerprint missing classifier weights")
    digest = hashlib.sha256()
    digest.update(
        json.dumps(asdict(parameters), sort_keys=True).encode("utf-8")
    )
    digest.update(str(state["embedding_dimension"]).encode("ascii"))
    for name, tensor in sorted(state["model_state"].items()):
        array = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(str(array.shape).encode("ascii"))
        digest.update(array.tobytes())
    return digest.hexdigest()


def classifier_fingerprint(classifier: BinaryEmbeddingClassifier):
    """Stable identity of the fitted weights and classifier configuration."""
    return fingerprint_classifier_state(classifier.parameters, classifier.export_state())


class DetectionScanner:
    """Score every embedding in a compatible corpus, saving positive matches.

    Unlike ``ActiveLearner``, this does not train, select review candidates, or
    exclude human-labelled refs. Human labels are independent of model scores.
    Call ``scan`` on a worker thread from the future DetectionsController.
    """

    def __init__(
        self, *, corpus: EmbeddingCorpus, classifier: BinaryEmbeddingClassifier,
        store: DetectionStore, classifier_id: str, batch_size: int = 8192,
    ):
        if not isinstance(corpus, EmbeddingCorpus):
            raise TypeError("corpus must be an EmbeddingCorpus")
        if not isinstance(classifier, BinaryEmbeddingClassifier):
            raise TypeError("classifier must be a BinaryEmbeddingClassifier")
        if not isinstance(store, DetectionStore):
            raise TypeError("store must be a DetectionStore")
        if int(batch_size) <= 0:
            raise ValueError("batch_size must be positive")
        if not str(classifier_id).strip():
            raise ValueError("classifier_id cannot be empty")
        self.corpus = corpus
        self.classifier = classifier
        self.store = store
        self.classifier_id = str(classifier_id).strip()
        self.batch_size = int(batch_size)

    def scan(self, *, threshold=0.5, progress_callback=None, cancel_event=None):
        """Return a completed ``DetectionRun``; failures leave no partial run.

        The threshold is a model-score cutoff, not a calibrated probability of
        the sound being present. Decreasing it later requires a new scan.
        """
        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise DetectionScanCancelled("Detection scan cancelled")

        check_cancelled()
        run = DetectionRun.create(
            classifier_id=self.classifier_id,
            model_id=classifier_fingerprint(self.classifier),
            embedding_space=self.corpus.space,
            threshold=threshold,
        )
        if progress_callback is not None:
            progress_callback(DetectionScanProgress(0, 0))

        with self.store.write_run(run) as writer:
            for batch in self.corpus.iter_batches(batch_size=self.batch_size):
                check_cancelled()
                probabilities = self.classifier.predict_proba(batch.vectors)
                check_cancelled()
                probabilities = np.asarray(probabilities)
                if probabilities.shape != (len(batch.refs),):
                    raise ValueError("Classifier returned the wrong number of scores")
                if not np.all(np.isfinite(probabilities)) or np.any(
                    (probabilities < 0.0) | (probabilities > 1.0)
                ):
                    raise ValueError("Classifier returned an invalid score")
                matches = (
                    Detection(run.run_id, ref, float(score))
                    for ref, score in zip(batch.refs, probabilities)
                    if float(score) >= run.threshold
                )
                check_cancelled()
                writer.add_batch(matches, scanned_count=len(batch))
                if progress_callback is not None:
                    progress_callback(
                        DetectionScanProgress(
                            writer.scored_count, writer.detection_count
                        )
                    )
            # Also check after the last batch, before the SQLite commit.
            check_cancelled()

        completed = self.store.get_run(run.run_id)
        if completed is None:
            raise RuntimeError("Completed detection run could not be reloaded")
        return completed
