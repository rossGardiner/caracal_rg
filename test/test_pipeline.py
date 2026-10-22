from unittest.mock import Mock

import pytest

from src.Pipeline import Pipeline
from src.PipelineLink import PipelineLink


class PassLink(PipelineLink):
    def configuration_parameters(self):
        return {}


class SourceLink(PassLink):
    def __init__(self, items):
        super().__init__()
        self.items = items

    def stream(self):
        for item in self.items:
            self.next_audio(item)
        return "stream-complete"


def test_empty_pipeline_has_no_links():
    pipeline = Pipeline()
    assert pipeline.links == ()
    assert pipeline.head is None
    assert pipeline.tail is None


@pytest.mark.parametrize(
    "operation", ["run", "next_audio", "get_config", "get_config_json", "get_config_hash"]
)
def test_empty_pipeline_operations_fail_clearly(operation):
    arguments = [object()] if operation == "next_audio" else []
    with pytest.raises(RuntimeError, match="empty Pipeline"):
        getattr(Pipeline(), operation)(*arguments)


def test_append_wires_links_and_exposes_an_immutable_snapshot():
    first, second = PassLink(), PassLink()
    pipeline = Pipeline([first])
    snapshot = pipeline.links

    assert pipeline.append(second) is second

    assert snapshot == (first,)
    assert pipeline.links == (first, second)
    assert pipeline.head is first
    assert pipeline.tail is second
    assert first.callback is second
    assert second.previous is first


@pytest.mark.parametrize("invalid", [None, object(), "link"])
def test_append_rejects_non_links_without_changing_the_graph(invalid):
    first = PassLink()
    pipeline = Pipeline([first])

    with pytest.raises(TypeError, match="PipelineLink"):
        pipeline.append(invalid)

    assert pipeline.links == (first,)
    assert first.callback is None


def test_next_audio_forwards_each_item_once_in_order():
    first, second = PassLink(), PassLink()
    pipeline = Pipeline([first, second])
    sink = Mock()
    second.register_callback(sink)
    items = [object(), object()]

    for item in items:
        pipeline.next_audio(item)

    assert [call.args[0] for call in sink.next_audio.call_args_list] == items


def test_terminal_link_without_a_callback_accepts_audio():
    assert Pipeline([PassLink()]).next_audio(object()) is None


def test_run_drives_a_source_and_returns_its_result():
    items = [object(), object()]
    source, tail = SourceLink(items), PassLink()
    pipeline = Pipeline([source, tail])
    sink = Mock()
    tail.register_callback(sink)

    assert pipeline.run() == "stream-complete"
    assert [call.args[0] for call in sink.next_audio.call_args_list] == items


def test_run_requires_a_source_driven_head():
    with pytest.raises(TypeError, match="stream"):
        Pipeline([PassLink()]).run()


def test_get_link_returns_the_unique_matching_instance():
    source, tail = SourceLink([]), PassLink()
    assert Pipeline([source, tail]).get_link(SourceLink) is source


def test_get_link_rejects_missing_or_ambiguous_matches():
    pipeline = Pipeline([PassLink(), PassLink()])
    with pytest.raises(LookupError, match="does not contain"):
        pipeline.get_link(SourceLink)
    with pytest.raises(LookupError, match="multiple"):
        pipeline.get_link(PassLink)


def test_configuration_methods_delegate_to_the_current_tail():
    first, second = PassLink(), PassLink()
    pipeline = Pipeline([first, second])

    assert pipeline.get_config() == second.get_config()
    assert pipeline.get_config_json() == second.get_config_json()
    assert pipeline.get_config_hash() == second.get_config_hash()
