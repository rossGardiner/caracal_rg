"""Shared pipeline definitions used by explorer and precompute entry points."""

from src.PipelineDefinition import PipelineDefinition, StageDefinition
from src.PipelineFactory import PipelineFactory


DEFAULT_DATA_ROOT = "/media/rossg/PortableSSD/BVC Sample Audio"
DEFAULT_MODEL_PATH = "assets/perch_v2.onnx"
DEFAULT_CACHE_DIRECTORY = "cache/embeddings"
DEFAULT_CHUNK_DURATION_S = 5.0


def _embedding_pipeline_definition(
    data_root=DEFAULT_DATA_ROOT,
    model_path=DEFAULT_MODEL_PATH,
    cache_directory=DEFAULT_CACHE_DIRECTORY,
    chunk_duration_s=DEFAULT_CHUNK_DURATION_S,
):
    return PipelineDefinition(
        name="Default embedding pipeline",
        description=(
            "Canonical CARACAL audio-to-Perch embedding pipeline. "
            "Compatible cached embeddings are identified by the runtime "
            "pipeline's existing configuration hash."
        ),
        stages=(
            StageDefinition(
                stage_type="caracal_streamer",
                label="CARACAL audio source",
                parameters={
                    "rootpath": data_root,
                },
            ),
            StageDefinition(
                stage_type="audio_buffer_loader",
                label="Canonical chunk loader",
                parameters={
                    "chunk_duration_s": chunk_duration_s,
                    "is_caracal": True,
                },
            ),
            StageDefinition(
                stage_type="high_pass_filter",
                label="High-pass filter",
                parameters={
                    "cutoff_hz": 60.0,
                    "order": 4,
                },
            ),
            StageDefinition(
                stage_type="speedometer",
                label="Speedometer",
                parameters={
                    "report_interval_s": 1.0,
                },
            ),
            StageDefinition(
                stage_type="resampler",
                label="Resampler",
                parameters={
                    "target_sample_rate": 32000,
                },
            ),
            StageDefinition(
                stage_type="embedding_cache_loader",
                label="Embedding cache lookup",
                parameters={
                    "cache_directory": cache_directory,
                },
            ),
            StageDefinition(
                stage_type="embeddings_creator",
                label="Perch v2 embeddings",
                parameters={
                    "model_path": model_path,
                    "use_cuda": True,
                    "embedding_name": "perch_v2",
                },
            ),
            StageDefinition(
                stage_type="embedding_cache_saver",
                label="Embedding cache save",
                parameters={
                    "cache_directory": cache_directory,
                },
            ),
        ),
    )


DEFAULT_EMBEDDING_PIPELINE = _embedding_pipeline_definition()


def build_embedding_pipeline(
    data_root=DEFAULT_DATA_ROOT,
    model_path=DEFAULT_MODEL_PATH,
    cache_directory=DEFAULT_CACHE_DIRECTORY,
    chunk_duration_s=DEFAULT_CHUNK_DURATION_S,
):
    """Build the canonical runtime pipeline from its editable definition.

    This compatibility helper preserves the old public construction function,
    but runtime links are now created solely by ``PipelineFactory`` from stage
    data rather than by a second hand-written callback graph.
    """

    definition = _embedding_pipeline_definition(
        data_root=data_root,
        model_path=model_path,
        cache_directory=cache_directory,
        chunk_duration_s=chunk_duration_s,
    )

    return PipelineFactory().build(definition)


def get_pipeline_definitions() -> tuple[PipelineDefinition, ...]:
    """Return the pipelines currently offered by the application."""

    return (
        DEFAULT_EMBEDDING_PIPELINE,
    )
