"""Construct runtime callback pipelines from editable pipeline definitions."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from src.AudioBufferLoader import AudioBufferLoader
from src.CanonicalChunkGrid import CanonicalChunkGrid
from src.CaracalStreamer import CaracalStreamer
from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.EmbeddingCacheSaver import EmbeddingCacheSaver
from src.EmbeddingsCreator import EmbeddingsCreator
from src.HighPassFilter import HighPassFilter
from src.Pipeline import Pipeline
from src.PipelineDefinition import PipelineDefinition, StageDefinition
from src.PipelineLink import PipelineLink
from src.Resampler import Resampler
from src.SpeedometerLink import SpeedometerLink


@dataclass(frozen=True)
class StageParameter:
    """Describe one editable parameter supported by a stage type."""

    name: str
    label: str
    kind: str
    default: Any
    minimum: float | int | None = None


@dataclass(frozen=True)
class StageType:
    """Registry entry describing how one stage type is edited and built."""

    stage_type: str
    label: str
    parameters: tuple[StageParameter, ...]
    builder: Callable[[dict[str, Any], "_BuildContext"], PipelineLink]

    def validated_parameters(
        self,
        stage: StageDefinition,
    ) -> dict[str, Any]:
        """Validate and complete one stage's parameter mapping."""

        specifications = {
            parameter.name: parameter
            for parameter in self.parameters
        }

        unknown = set(stage.parameters) - set(specifications)
        if unknown:
            names = ", ".join(sorted(unknown))
            raise ValueError(
                f"Unknown parameter(s) for {self.stage_type}: {names}"
            )

        values = {}
        for parameter in self.parameters:
            value = stage.parameters.get(
                parameter.name,
                parameter.default,
            )
            values[parameter.name] = _validate_parameter_value(
                parameter,
                value,
            )

        return values


class PipelineFactory:
    """Build fresh ``Pipeline`` instances from ``PipelineDefinition`` data."""

    def __init__(
        self,
        stage_types: tuple[StageType, ...] | None = None,
    ):
        if stage_types is None:
            stage_types = DEFAULT_STAGE_TYPES

        self._stage_types = {
            stage_type.stage_type: stage_type
            for stage_type in stage_types
        }

        if len(self._stage_types) != len(stage_types):
            raise ValueError("Stage type identifiers must be unique")

    @property
    def stage_types(self) -> tuple[StageType, ...]:
        """Return the registered stage types in their declared order."""

        return tuple(self._stage_types.values())

    def get_stage_type(
        self,
        stage_type: str,
    ) -> StageType:
        try:
            return self._stage_types[stage_type]
        except KeyError as exc:
            raise ValueError(
                f"Unknown pipeline stage type: {stage_type}"
            ) from exc

    def build(
        self,
        definition: PipelineDefinition,
    ) -> Pipeline:
        """Construct a fresh callback graph from definition data."""

        if not isinstance(definition, PipelineDefinition):
            raise TypeError("definition must be a PipelineDefinition")

        context = _BuildContext(
            factory=self,
            definition=definition,
        )

        links = [
            context.build_stage(index)
            for index in range(len(definition.stages))
        ]

        return Pipeline(links)


class _BuildContext:
    """Resolve stage dependencies while a single pipeline is constructed."""

    def __init__(
        self,
        factory: PipelineFactory,
        definition: PipelineDefinition,
    ):
        self.factory = factory
        self.definition = definition
        self._instances: dict[int, PipelineLink] = {}
        self._building: set[int] = set()
        self._services: dict[tuple[Any, ...], Any] = {}

    def build_stage(
        self,
        index: int,
    ) -> PipelineLink:
        if index in self._instances:
            return self._instances[index]

        if index in self._building:
            stage = self.definition.stages[index]
            raise ValueError(
                f"Circular pipeline-stage dependency involving {stage.stage_type}"
            )

        stage = self.definition.stages[index]
        stage_type = self.factory.get_stage_type(stage.stage_type)
        parameters = stage_type.validated_parameters(stage)

        self._building.add(index)
        try:
            link = stage_type.builder(parameters, self)
        finally:
            self._building.remove(index)

        if not isinstance(link, PipelineLink):
            raise TypeError(
                f"Stage builder for {stage.stage_type} did not return a PipelineLink"
            )

        self._instances[index] = link
        return link

    def require_stage(
        self,
        stage_type: str,
    ) -> PipelineLink:
        """Return the unique runtime link with ``stage_type`` in this definition."""

        matches = [
            index
            for index, stage in enumerate(self.definition.stages)
            if stage.stage_type == stage_type
        ]

        if not matches:
            raise ValueError(
                f"Pipeline requires a {stage_type} stage"
            )

        if len(matches) > 1:
            raise ValueError(
                f"Pipeline contains multiple {stage_type} stages; dependency is ambiguous"
            )

        return self.build_stage(matches[0])

    def service(
        self,
        key: tuple[Any, ...],
        builder: Callable[[], Any],
    ) -> Any:
        """Return one shared non-link service for this pipeline build."""

        if key not in self._services:
            self._services[key] = builder()

        return self._services[key]


def _embedding_cache(
    context: _BuildContext,
    root_directory: str,
) -> EmbeddingCache:
    return context.service(
        ("embedding_cache", root_directory),
        lambda: EmbeddingCache(root_directory=root_directory),
    )


def _build_caracal_streamer(parameters, context):
    return CaracalStreamer(
        rootpath=parameters["rootpath"],
    )


def _build_audio_buffer_loader(parameters, context):
    return AudioBufferLoader(
        chunk_grid=CanonicalChunkGrid(
            chunk_duration_s=parameters["chunk_duration_s"],
        ),
        is_caracal=parameters["is_caracal"],
    )


def _build_high_pass_filter(parameters, context):
    return HighPassFilter(
        cutoff_hz=parameters["cutoff_hz"],
        order=parameters["order"],
    )


def _build_speedometer(parameters, context):
    return SpeedometerLink(
        report_interval_s=parameters["report_interval_s"],
    )


def _build_resampler(parameters, context):
    return Resampler(
        target_sample_rate=parameters["target_sample_rate"],
    )


def _build_embedding_cache_loader(parameters, context):
    embeddings_creator = context.require_stage(
        "embeddings_creator"
    )

    if not isinstance(embeddings_creator, EmbeddingsCreator):
        raise TypeError(
            "embeddings_creator stage must build an EmbeddingsCreator"
        )

    return EmbeddingCacheLoader(
        cache=_embedding_cache(
            context,
            parameters["cache_directory"],
        ),
        embeddings_creator=embeddings_creator,
    )


def _build_embeddings_creator(parameters, context):
    return EmbeddingsCreator(
        model_path=parameters["model_path"],
        use_cuda=parameters["use_cuda"],
        embedding_name=parameters["embedding_name"],
    )


def _build_embedding_cache_saver(parameters, context):
    embeddings_creator = context.require_stage(
        "embeddings_creator"
    )

    if not isinstance(embeddings_creator, EmbeddingsCreator):
        raise TypeError(
            "embeddings_creator stage must build an EmbeddingsCreator"
        )

    return EmbeddingCacheSaver(
        cache=_embedding_cache(
            context,
            parameters["cache_directory"],
        ),
        embeddings_creator=embeddings_creator,
    )


def _validate_parameter_value(
    parameter: StageParameter,
    value: Any,
) -> Any:
    kind = parameter.kind

    if kind == "boolean":
        if not isinstance(value, bool):
            raise TypeError(f"{parameter.name} must be a boolean")
        validated = value

    elif kind == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError(f"{parameter.name} must be an integer")
        validated = value

    elif kind == "number":
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"{parameter.name} must be a number")
        validated = float(value)

    elif kind in {"text", "file", "directory"}:
        if not isinstance(value, str):
            raise TypeError(f"{parameter.name} must be text")
        validated = value

    else:
        raise ValueError(
            f"Unsupported parameter kind {kind!r} for {parameter.name}"
        )

    if (
        parameter.minimum is not None
        and isinstance(validated, (int, float))
        and not isinstance(validated, bool)
        and validated < parameter.minimum
    ):
        raise ValueError(
            f"{parameter.name} must be at least {parameter.minimum}"
        )

    return validated


DEFAULT_STAGE_TYPES = (
    StageType(
        stage_type="caracal_streamer",
        label="CARACAL audio source",
        parameters=(
            StageParameter(
                name="rootpath",
                label="Data root",
                kind="directory",
                default=".",
            ),
        ),
        builder=_build_caracal_streamer,
    ),
    StageType(
        stage_type="audio_buffer_loader",
        label="Canonical chunk loader",
        parameters=(
            StageParameter(
                name="chunk_duration_s",
                label="Chunk duration (s)",
                kind="number",
                default=5.0,
                minimum=0.001,
            ),
            StageParameter(
                name="is_caracal",
                label="CARACAL input",
                kind="boolean",
                default=True,
            ),
        ),
        builder=_build_audio_buffer_loader,
    ),
    StageType(
        stage_type="high_pass_filter",
        label="High-pass filter",
        parameters=(
            StageParameter(
                name="cutoff_hz",
                label="Cutoff (Hz)",
                kind="number",
                default=60.0,
                minimum=0.001,
            ),
            StageParameter(
                name="order",
                label="Filter order",
                kind="integer",
                default=4,
                minimum=1,
            ),
        ),
        builder=_build_high_pass_filter,
    ),
    StageType(
        stage_type="speedometer",
        label="Speedometer",
        parameters=(
            StageParameter(
                name="report_interval_s",
                label="Report interval (s)",
                kind="number",
                default=1.0,
                minimum=0.001,
            ),
        ),
        builder=_build_speedometer,
    ),
    StageType(
        stage_type="resampler",
        label="Resampler",
        parameters=(
            StageParameter(
                name="target_sample_rate",
                label="Target sample rate (Hz)",
                kind="integer",
                default=32000,
                minimum=1,
            ),
        ),
        builder=_build_resampler,
    ),
    StageType(
        stage_type="embedding_cache_loader",
        label="Embedding cache lookup",
        parameters=(
            StageParameter(
                name="cache_directory",
                label="Cache directory",
                kind="directory",
                default="cache/embeddings",
            ),
        ),
        builder=_build_embedding_cache_loader,
    ),
    StageType(
        stage_type="embeddings_creator",
        label="Perch v2 embeddings",
        parameters=(
            StageParameter(
                name="model_path",
                label="Model file",
                kind="file",
                default="assets/perch_v2.onnx",
            ),
            StageParameter(
                name="use_cuda",
                label="Use CUDA when available",
                kind="boolean",
                default=True,
            ),
            StageParameter(
                name="embedding_name",
                label="Embedding name",
                kind="text",
                default="perch_v2",
            ),
        ),
        builder=_build_embeddings_creator,
    ),
    StageType(
        stage_type="embedding_cache_saver",
        label="Embedding cache save",
        parameters=(
            StageParameter(
                name="cache_directory",
                label="Cache directory",
                kind="directory",
                default="cache/embeddings",
            ),
        ),
        builder=_build_embedding_cache_saver,
    ),
)
