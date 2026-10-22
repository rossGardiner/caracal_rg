"""Opt-in real ONNX imports with a fake session boundary, not model inference."""

import threading
from unittest.mock import Mock

import onnxruntime as ort
import pytest

from src.EmbeddingCache import EmbeddingCache
from src.EmbeddingCacheLoader import EmbeddingCacheLoader
from src.EmbeddingCacheSaver import EmbeddingCacheSaver
from src.EmbeddingsCreator import EmbeddingsCreator, _EmbeddingRuntime
from src.HighPassFilter import HighPassFilter
from src.Pipeline import Pipeline
from src.PipelineDefinition import PipelineDefinition, StageDefinition
from src.PipelineFactory import PipelineFactory, StageType
from src.PipelineLink import PipelineLink
from src.Resampler import Resampler
from src.SpeedometerLink import SpeedometerLink


@pytest.fixture
def runtime(monkeypatch):
    # Real application classes and the real ONNX package are imported above.
    # Guard against accidental model construction; no external model is needed.
    session_constructor = Mock(side_effect=AssertionError("Must not load a model session"))
    monkeypatch.setattr(ort, "InferenceSession", session_constructor)
    result = _EmbeddingRuntime(
        session=Mock(spec=["run"]), input_name="audio", input_shape=[1, 32],
        input_type="tensor(float)", inference_lock=threading.Lock(),
    )
    yield result
    session_constructor.assert_not_called()


def test_real_cache_and_telemetry_links_preserve_processing_hashes(tmp_path, runtime):
    creator = EmbeddingsCreator("fixture.onnx", use_cuda=False, _shared_runtime=runtime)
    baseline = Pipeline([HighPassFilter(), Resampler(), creator])
    original_json, original_hash = baseline.get_config_json(), baseline.get_config_hash()
    other_creator = creator.shared_runtime_link()
    cache = EmbeddingCache(tmp_path / "cache")
    observed = Pipeline([
        HighPassFilter(), SpeedometerLink(), Resampler(),
        EmbeddingCacheLoader(cache, other_creator), other_creator,
        EmbeddingCacheSaver(cache, other_creator),
    ])

    assert observed.get_config_json() == original_json
    assert observed.get_config_hash() == original_hash
    assert other_creator.get_config_hash() == original_hash
    assert baseline.get_config_hash() == original_hash


def test_builtin_factory_wires_shared_creator_and_cache_without_loading_models(
    tmp_path, monkeypatch, runtime
):
    create_runtime = Mock(return_value=runtime)
    monkeypatch.setattr(EmbeddingsCreator, "_create_runtime", staticmethod(create_runtime))
    factory = PipelineFactory()
    cache_root = str(tmp_path / "cache")
    candidate = PipelineDefinition("Built-in links", "", (
        StageDefinition("audio_buffer_loader", "Load", {"is_caracal": False}),
        StageDefinition("high_pass_filter", "Filter"),
        StageDefinition("resampler", "Resample"),
        StageDefinition("embedding_cache_loader", "Cache lookup", {"cache_directory": cache_root}),
        StageDefinition("embeddings_creator", "Model", {
            "model_path": "fixture.onnx", "use_cuda": False, "embedding_name": "fixture-model"
        }),
        StageDefinition("embedding_cache_saver", "Cache save", {"cache_directory": cache_root}),
    ))

    first, second = factory.build(candidate), factory.build(candidate)

    assert create_runtime.call_count == 2
    create_runtime.assert_called_with(model_path="fixture.onnx", use_cuda=False)
    for pipeline in [first, second]:
        loader = pipeline.get_link(EmbeddingCacheLoader)
        saver = pipeline.get_link(EmbeddingCacheSaver)
        creator = pipeline.get_link(EmbeddingsCreator)
        assert loader.embeddings_creator is saver.embeddings_creator is creator
        assert loader.cache is saver.cache
        assert creator.embedding_name == "fixture-model"
        assert creator.get_config_hash() == pipeline.get_config_hash()
    assert first.get_link(EmbeddingsCreator) is not second.get_link(EmbeddingsCreator)
    assert first.get_link(EmbeddingCacheLoader).cache is not second.get_link(EmbeddingCacheLoader).cache
    assert first.get_config_hash() == second.get_config_hash()


@pytest.mark.parametrize("cache_stage", ["embedding_cache_loader", "embedding_cache_saver"])
@pytest.mark.parametrize("creator_count", [0, 2])
def test_builtin_cache_dependencies_must_be_present_and_unambiguous(
    cache_stage, creator_count, runtime
):
    factory = PipelineFactory()
    stages = (factory.get_stage_type(cache_stage).default_stage(),) + tuple(
        factory.get_stage_type("embeddings_creator").default_stage() for _ in range(creator_count)
    )
    with pytest.raises(ValueError, match="requires|ambiguous"):
        factory.build(PipelineDefinition("Invalid", "", stages))


class NonEmbeddingLink(PipelineLink):
    def configuration_parameters(self):
        return {}


@pytest.mark.parametrize("cache_stage", ["embedding_cache_loader", "embedding_cache_saver"])
def test_builtin_cache_builder_rejects_a_wrong_creator_type(cache_stage, runtime):
    cache_type = PipelineFactory().get_stage_type(cache_stage)
    wrong_creator = StageType(
        "embeddings_creator", "Wrong creator", (), lambda parameters, context: NonEmbeddingLink()
    )
    factory = PipelineFactory((cache_type, wrong_creator))

    with pytest.raises(TypeError, match="must build an EmbeddingsCreator"):
        factory.build(PipelineDefinition("Invalid", "", (
            cache_type.default_stage(), wrong_creator.default_stage()
        )))
