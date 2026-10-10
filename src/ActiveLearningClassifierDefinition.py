"""Durable definition for one named active-learning classifier."""

from dataclasses import asdict, dataclass, replace
from uuid import uuid4

from src.ActiveLearner import ActiveLearningParameters, BinaryLabel
from src.BinaryEmbeddingClassifier import BinaryClassifierParameters
from src.EmbeddingCorpus import EmbeddingRef, EmbeddingSpaceKey


@dataclass(frozen=True, order=True)
class LabelledEmbedding:
    """One durable human label attached to an embedding reference."""

    ref: EmbeddingRef
    label: BinaryLabel

    def __post_init__(self):
        if not isinstance(self.ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef")
        if not isinstance(self.label, BinaryLabel):
            raise TypeError("label must be a BinaryLabel")

    def to_dict(self):
        return {
            "recording_id": self.ref.recording_id,
            "chunk_index": self.ref.chunk_index,
            "label": self.label.name,
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("Labelled embedding must be an object")

        try:
            label = BinaryLabel[str(payload["label"])]
            ref = EmbeddingRef(
                recording_id=str(payload["recording_id"]),
                chunk_index=int(payload["chunk_index"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid labelled embedding") from exc

        return cls(ref=ref, label=label)


@dataclass(frozen=True)
class ActiveLearningClassifierDefinition:
    """Persistent configuration and human labels for one binary detector.

    This object deliberately contains no corpus, PyTorch module, review queue,
    cache object, or Qt state.  Those are runtime concerns.  The durable
    binding is the ``EmbeddingSpaceKey``; ``pipeline_name`` is only the last
    pipeline definition used to resolve that space for the UI.
    """

    classifier_id: str
    name: str
    pipeline_name: str | None = None
    embedding_space: EmbeddingSpaceKey | None = None
    classifier_parameters: BinaryClassifierParameters = BinaryClassifierParameters()
    active_learning_parameters: ActiveLearningParameters = ActiveLearningParameters()
    labels: tuple[LabelledEmbedding, ...] = ()

    def __post_init__(self):
        classifier_id = str(self.classifier_id).strip()
        name = str(self.name).strip()
        if not classifier_id:
            raise ValueError("classifier_id cannot be empty")
        if not name:
            raise ValueError("classifier name cannot be empty")

        pipeline_name = self.pipeline_name
        if pipeline_name is not None:
            pipeline_name = str(pipeline_name).strip()
            if not pipeline_name:
                pipeline_name = None

        if self.embedding_space is not None and not isinstance(
            self.embedding_space,
            EmbeddingSpaceKey,
        ):
            raise TypeError("embedding_space must be an EmbeddingSpaceKey or None")
        if not isinstance(self.classifier_parameters, BinaryClassifierParameters):
            raise TypeError(
                "classifier_parameters must be BinaryClassifierParameters"
            )
        if not isinstance(self.active_learning_parameters, ActiveLearningParameters):
            raise TypeError(
                "active_learning_parameters must be ActiveLearningParameters"
            )

        labels = tuple(self.labels)
        if not all(isinstance(item, LabelledEmbedding) for item in labels):
            raise TypeError("labels must contain LabelledEmbedding objects")

        refs = tuple(item.ref for item in labels)
        if len(set(refs)) != len(refs):
            raise ValueError("A classifier cannot contain duplicate labelled refs")

        object.__setattr__(self, "classifier_id", classifier_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "pipeline_name", pipeline_name)
        object.__setattr__(self, "labels", tuple(sorted(labels)))

    @classmethod
    def create(cls, name: str):
        return cls(
            classifier_id=uuid4().hex,
            name=name,
        )

    @property
    def positive_count(self):
        return sum(item.label is BinaryLabel.POSITIVE for item in self.labels)

    @property
    def negative_count(self):
        return sum(item.label is BinaryLabel.NEGATIVE for item in self.labels)

    def label_for(self, ref: EmbeddingRef):
        if not isinstance(ref, EmbeddingRef):
            raise TypeError("ref must be an EmbeddingRef")
        for item in self.labels:
            if item.ref == ref:
                return item.label
        return None

    def with_binding(
        self,
        *,
        pipeline_name: str,
        embedding_space: EmbeddingSpaceKey,
    ):
        """Bind once to an embedding space; equivalent pipelines may be reused."""

        if not isinstance(embedding_space, EmbeddingSpaceKey):
            raise TypeError("embedding_space must be an EmbeddingSpaceKey")

        if (
            self.embedding_space is not None
            and self.embedding_space != embedding_space
        ):
            raise ValueError(
                "Classifier is already bound to a different embedding space"
            )

        return replace(
            self,
            pipeline_name=str(pipeline_name),
            embedding_space=embedding_space,
        )

    def with_labels_from(self, learner):
        """Return a copy containing the learner's current human labels."""

        labelled = [
            LabelledEmbedding(ref=ref, label=BinaryLabel.POSITIVE)
            for ref in learner.positive_refs
        ]
        labelled.extend(
            LabelledEmbedding(ref=ref, label=BinaryLabel.NEGATIVE)
            for ref in learner.negative_refs
        )
        return replace(self, labels=tuple(labelled))

    def to_dict(self):
        space = None
        if self.embedding_space is not None:
            space = {
                "pipeline_hash": self.embedding_space.pipeline_hash,
                "embedding_name": self.embedding_space.embedding_name,
            }

        return {
            "classifier_id": self.classifier_id,
            "name": self.name,
            "pipeline_name": self.pipeline_name,
            "embedding_space": space,
            "classifier_parameters": asdict(self.classifier_parameters),
            "active_learning_parameters": asdict(
                self.active_learning_parameters
            ),
            "labels": [item.to_dict() for item in self.labels],
        }

    @classmethod
    def from_dict(cls, payload):
        if not isinstance(payload, dict):
            raise TypeError("Classifier definition must be an object")

        space_payload = payload.get("embedding_space")
        if space_payload is None:
            space = None
        else:
            if not isinstance(space_payload, dict):
                raise ValueError("embedding_space must be an object or null")
            try:
                space = EmbeddingSpaceKey(
                    pipeline_hash=str(space_payload["pipeline_hash"]),
                    embedding_name=str(space_payload["embedding_name"]),
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError("Invalid embedding_space") from exc

        try:
            classifier_parameters = BinaryClassifierParameters(
                **payload.get("classifier_parameters", {})
            )
            active_learning_parameters = ActiveLearningParameters(
                **payload.get("active_learning_parameters", {})
            )
            labels = tuple(
                LabelledEmbedding.from_dict(item)
                for item in payload.get("labels", [])
            )
            return cls(
                classifier_id=str(payload["classifier_id"]),
                name=str(payload["name"]),
                pipeline_name=payload.get("pipeline_name"),
                embedding_space=space,
                classifier_parameters=classifier_parameters,
                active_learning_parameters=active_learning_parameters,
                labels=labels,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("Invalid classifier definition") from exc
