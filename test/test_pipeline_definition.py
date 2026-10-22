import pytest

from src.PipelineDefinition import PipelineDefinition, StageDefinition


def test_stage_normalizes_names_and_copies_the_parameter_mapping():
    parameters = {"factor": 2}
    stage = StageDefinition(" gain ", " Gain ", parameters)
    parameters["factor"] = 99

    assert stage.to_dict() == {
        "type": "gain", "label": "Gain", "parameters": {"factor": 2}
    }


def test_stage_defaults_do_not_share_mutable_parameters():
    first = StageDefinition("gain", "Gain")
    second = StageDefinition("gain", "Gain")
    first.parameters["factor"] = 2
    assert second.parameters == {}


@pytest.mark.parametrize("stage_type, label", [("", "Gain"), ("  ", "Gain"), ("gain", " ")])
def test_stage_requires_nonempty_type_and_label(stage_type, label):
    with pytest.raises(ValueError, match="cannot be empty"):
        StageDefinition(stage_type, label)


def test_stage_rejects_non_json_parameter_values():
    with pytest.raises(TypeError, match="JSON serialisable"):
        StageDefinition("gain", "Gain", {"bad": object()})


def test_stage_rejects_circular_parameters():
    cycle = []
    cycle.append(cycle)
    with pytest.raises(TypeError, match="JSON serialisable"):
        StageDefinition("gain", "Gain", {"cycle": cycle})


def test_stage_from_dict_defaults_label_and_parameters():
    assert StageDefinition.from_dict({"type": "gain"}).to_dict() == {
        "type": "gain", "label": "gain", "parameters": {}
    }


def test_definition_roundtrip_preserves_stage_order_and_labels():
    data = {
        "name": "Example", "description": "Two stages",
        "stages": [
            {"type": "gain", "label": "First", "parameters": {"factor": 2}},
            {"type": "gain", "label": "Second", "parameters": {"factor": 3}},
        ],
    }
    definition = PipelineDefinition.from_dict(data)

    assert definition.to_dict() == data
    assert definition.stage_names == ("First", "Second")
    assert isinstance(definition.stages, tuple)


def test_definition_normalizes_name_and_defaults_description_when_read():
    definition = PipelineDefinition.from_dict({"name": " Example ", "stages": [{"type": "gain"}]})
    assert definition.name == "Example"
    assert definition.description == ""


@pytest.mark.parametrize("name", ["", "   "])
def test_definition_requires_a_name(name):
    with pytest.raises(ValueError, match="name cannot be empty"):
        PipelineDefinition(name, "", (StageDefinition("gain", "Gain"),))


def test_definition_requires_at_least_one_stage():
    with pytest.raises(ValueError, match="at least one stage"):
        PipelineDefinition("Example", "", ())


def test_definition_requires_stage_definition_objects():
    with pytest.raises(TypeError, match="StageDefinition"):
        PipelineDefinition("Example", "", ({"type": "gain"},))


@pytest.mark.parametrize("name", [None, "Copy"])
def test_copy_is_detached_including_nested_stage_parameters(name):
    original = PipelineDefinition(
        "Original", "Description",
        (StageDefinition("gain", "Gain", {"nested": {"weights": [1, 2]}}),),
    )

    duplicate = original.copy(name=name)
    duplicate.stages[0].parameters["nested"]["weights"].append(3)
    duplicate.stages[0].label = "Edited"

    assert duplicate.name == ("Original" if name is None else "Copy")
    assert duplicate.stages[0] is not original.stages[0]
    assert original.stages[0].parameters == {"nested": {"weights": [1, 2]}}
    assert original.stages[0].label == "Gain"
