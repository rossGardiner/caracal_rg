from pathlib import Path
import subprocess
import sys
from unittest.mock import Mock

import pytest

from src.PipelineDefinition import PipelineDefinition, StageDefinition
from src.PipelineFactory import PipelineFactory, StageParameter, StageType
from src.PipelineLink import PipelineLink


class ParameterLink(PipelineLink):
    def __init__(self, parameters):
        super().__init__()
        self.parameters = dict(parameters)

    def configuration_parameters(self):
        return self.parameters


def stage(kind="custom", **parameters):
    return StageDefinition(kind, kind, parameters)


def definition(*stages):
    return PipelineDefinition("Example", "Factory fixture", stages)


def stage_type(kind="number", default=1, minimum=None, builder=None, name="custom"):
    return StageType(
        name, name,
        (StageParameter("value", "Value", kind, default, minimum),),
        builder if builder is not None else lambda parameters, context: ParameterLink(parameters),
    )


def test_custom_registry_and_metadata_do_not_import_unused_runtime_integrations():
    # Deny optional imports in a fresh interpreter; do not replace their APIs.
    code = '''
import importlib.abc
import sys
class DenyOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'caracal', 'onnxruntime'}:
            raise ImportError('Blocked runtime integration: ' + fullname)
sys.meta_path.insert(0, DenyOptional())
from src.PipelineFactory import PipelineFactory, StageType
from src.PipelineDefinition import PipelineDefinition, StageDefinition
from src.PipelineLink import PipelineLink
class LocalLink(PipelineLink):
    def configuration_parameters(self):
        return {}
custom = StageType('local', 'Local', (), lambda parameters, context: LocalLink())
factory = PipelineFactory((custom,))
pipeline = factory.build(PipelineDefinition('Local', '', (custom.default_stage(),)))
assert isinstance(pipeline.head, LocalLink)
for kind in ['caracal_streamer', 'embeddings_creator']:
    default = PipelineFactory().get_stage_type(kind).default_stage()
    try:
        PipelineFactory().build(PipelineDefinition('Needs integration', '', (default,)))
    except ImportError as error:
        assert 'Blocked runtime integration' in str(error)
    else:
        raise AssertionError('Selected integration must still require its dependency')
'''
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[1],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_registry_preserves_order_and_rejects_duplicate_identifiers():
    first, second = stage_type(name="first"), stage_type(name="second")
    factory = PipelineFactory((first, second))
    assert factory.stage_types == (first, second)
    assert factory.get_stage_type("second") is second
    assert PipelineFactory(()).stage_types == ()
    with pytest.raises(ValueError, match="unique"):
        PipelineFactory((first, first))
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        factory.get_stage_type("missing")


def test_default_stage_and_validated_defaults_have_independent_parameter_maps():
    specification = stage_type(default=2)
    first, second = specification.default_stage(), specification.default_stage()
    first.parameters["value"] = 99
    assert second.parameters == {"value": 2}
    assert specification.validated_parameters(stage()) == {"value": 2.0}


@pytest.mark.parametrize(
    "kind, value, expected",
    [("boolean", True, True), ("integer", 3, 3), ("number", 3, 3.0),
     ("number", 2.5, 2.5), ("text", "bird", "bird"),
     ("file", "assets/model.onnx", "assets/model.onnx"), ("directory", "audio", "audio")],
)
def test_supported_parameter_kinds_are_validated(kind, value, expected):
    result = stage_type(kind=kind).validated_parameters(stage(value=value))["value"]
    assert result == expected
    assert type(result) is type(expected)


@pytest.mark.parametrize(
    "kind, value",
    [("boolean", 1), ("integer", True), ("integer", 1.5), ("integer", "2"),
     ("number", False), ("number", "2"), ("text", None),
     ("file", Path("model.onnx")), ("directory", False)],
)
def test_parameter_type_errors_fail_before_construction(kind, value):
    # Path is intentionally introduced after the serialisable definition exists.
    candidate = stage(value=0)
    candidate.parameters["value"] = value
    with pytest.raises(TypeError):
        stage_type(kind=kind).validated_parameters(candidate)


@pytest.mark.parametrize("kind, value", [("integer", 0), ("number", 0.5)])
def test_numeric_minimum_is_enforced(kind, value):
    with pytest.raises(ValueError, match="at least"):
        stage_type(kind=kind, minimum=1).validated_parameters(stage(value=value))
    assert stage_type(kind=kind, minimum=1).validated_parameters(stage(value=1))["value"] == 1


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_number_parameters_must_be_finite(value):
    builder = Mock(return_value=ParameterLink({}))
    factory = PipelineFactory((stage_type(kind="number", builder=builder),))
    with pytest.raises(ValueError, match="finite"):
        factory.build(definition(stage(value=value)))
    builder.assert_not_called()


def test_unknown_parameters_and_unsupported_kinds_are_rejected():
    with pytest.raises(ValueError, match="alpha, beta"):
        stage_type().validated_parameters(stage(beta=1, alpha=2))
    with pytest.raises(ValueError, match="Unsupported parameter kind"):
        stage_type(kind="choice").validated_parameters(stage(value="a"))


def test_factory_rejects_invalid_definitions_before_calling_any_builder():
    builder = Mock(return_value=ParameterLink({}))
    factory = PipelineFactory((stage_type(builder=builder),))
    with pytest.raises(TypeError, match="PipelineDefinition"):
        factory.build(object())
    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        factory.build(definition(stage(), stage("missing")))
    with pytest.raises(TypeError):
        factory.build(definition(stage(), stage(value=True)))
    builder.assert_not_called()


def test_builds_resolve_forward_dependencies_once_and_scope_services_per_build():
    producer = Mock(side_effect=lambda parameters, context: ParameterLink({"role": "producer"}))
    resource = Mock(side_effect=object)

    def consume(parameters, context):
        link = ParameterLink({"role": "consumer"})
        link.dependency = context.require_stage("producer")
        link.resource = context.service(("resource",), resource)
        assert context.service(("resource",), resource) is link.resource
        return link

    factory = PipelineFactory((
        StageType("consumer", "Consumer", (), consume),
        StageType("producer", "Producer", (), producer),
    ))
    candidate = definition(stage("consumer"), stage("producer"), stage("consumer"))
    first, second = factory.build(candidate), factory.build(candidate)

    assert producer.call_count == resource.call_count == 2
    for pipeline in [first, second]:
        left, dependency, right = pipeline.links
        assert left.dependency is right.dependency is dependency
        assert left.resource is right.resource
        assert left.callback is dependency and dependency.callback is right
    assert first.links[0].resource is not second.links[0].resource
    assert all(left is not right for left, right in zip(first.links, second.links))


@pytest.mark.parametrize("producer_count", [0, 2])
def test_required_stages_must_be_present_and_unambiguous(producer_count):
    consumer = StageType("consumer", "Consumer", (), lambda p, c: c.require_stage("producer"))
    producer = StageType("producer", "Producer", (), lambda p, c: ParameterLink({}))
    factory = PipelineFactory((consumer, producer))
    candidate = definition(stage("consumer"), *[stage("producer") for _ in range(producer_count)])
    with pytest.raises(ValueError, match="requires|ambiguous"):
        factory.build(candidate)


def test_circular_stage_dependencies_raise_a_clear_error():
    first = StageType("first", "First", (), lambda p, c: c.require_stage("second"))
    second = StageType("second", "Second", (), lambda p, c: c.require_stage("first"))
    with pytest.raises(ValueError, match="Circular"):
        PipelineFactory((first, second)).build(definition(stage("first"), stage("second")))


def test_builder_must_return_a_pipeline_link():
    invalid = stage_type(builder=lambda parameters, context: object())
    with pytest.raises(TypeError, match="PipelineLink"):
        PipelineFactory((invalid,)).build(definition(stage()))


def test_builder_failure_propagates_and_does_not_poison_the_next_build():
    builder = Mock(side_effect=[RuntimeError("broken builder"), ParameterLink({})])
    factory = PipelineFactory((stage_type(builder=builder),))
    with pytest.raises(RuntimeError, match="broken builder"):
        factory.build(definition(stage()))
    assert isinstance(factory.build(definition(stage())).head, ParameterLink)


def test_builtin_dsp_stages_receive_their_validated_parameters():
    pipeline = PipelineFactory().build(definition(
        stage("high_pass_filter", cutoff_hz=90, order=2),
        stage("speedometer", report_interval_s=10),
        stage("resampler", target_sample_rate=22050),
    ))
    highpass, speed, resampler = pipeline.links
    assert highpass.configuration_parameters() == {"cutoff_hz": 90.0, "order": 2}
    assert speed.report_interval_s == 10.0
    assert resampler.configuration_parameters() == {"target_sample_rate": 22050}
    assert [item["link_type"] for item in pipeline.get_config()["links"]] == [
        "HighPassFilter", "Resampler"
    ]


def test_loader_keeps_the_legacy_buffer_seconds_configuration_key():
    pipeline = PipelineFactory().build(definition(
        stage("audio_buffer_loader", chunk_duration_s=0.25, is_caracal=False)
    ))
    assert pipeline.head.configuration_json() == (
        '{"link_type":"AudioBufferLoader","parameters":'
        '{"buffer_seconds":0.25,"is_caracal":false},"version":1}'
    )
