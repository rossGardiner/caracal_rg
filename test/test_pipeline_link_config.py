import json
import hashlib
import pytest

from src.Pipeline import Pipeline
from src.PipelineLink import PipelineLink
from src.SpeedometerLink import SpeedometerLink

class ConfigLink(PipelineLink):
    def __init__(self, params):
        super().__init__()
        self._params = params

    def configuration_parameters(self) -> dict[str, any]:
        return self._params


def test_configuration_serialization_and_hash():
    link = ConfigLink({"alpha": 1, "beta": "x"})
    cfg = link.configuration()
    assert cfg["link_type"] == "ConfigLink"
    assert "version" in cfg and "parameters" in cfg

    cfg_json = link.configuration_json()
    assert json.loads(cfg_json) == cfg

    expected_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
    assert link.configuration_hash() == expected_hash


def test_get_config_links_respects_chain_order():
    a = ConfigLink({"a": 1})
    b = ConfigLink({"b": 2})
    c = ConfigLink({"c": 3})

    a.register_callback(b)
    b.register_callback(c)

    links = c.get_config_links()
    assert [l["link_type"] for l in links] == ["ConfigLink", "ConfigLink", "ConfigLink"]
    assert [l["parameters"] for l in links] == [{"a":1}, {"b":2}, {"c":3}]

    # Also verify json and hash on the aggregated config
    cfg = c.get_config()
    cfg_json = c.get_config_json()
    assert json.loads(cfg_json) == cfg
    expected_agg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
    assert c.get_config_hash() == expected_agg_hash


def test_known_configuration_has_fixed_serialization_and_hash():
    link = ConfigLink({"beta": "x", "alpha": 1})
    expected = '{"link_type":"ConfigLink","parameters":{"alpha":1,"beta":"x"},"version":1}'

    assert link.configuration_json() == expected
    assert link.configuration_hash() == (
        "088d0a79aff2cc9b03f273bf39769dfe47632b0765a6cf01d0e16379e8d861d6"
    )
    assert link.get_config_json() == '{"links":[' + expected + '],"schema_version":1}'
    assert link.get_config_hash() == "372d607fc967a1bb1a56dc4b76a28f6daa78c6a26e8e21e6a5ccda12fcbcca1d"


def test_parameter_insertion_order_does_not_change_compatibility_hashes():
    first = ConfigLink({"alpha": 1, "nested": {"a": 2, "b": 3}})
    second = ConfigLink({"nested": {"b": 3, "a": 2}, "alpha": 1})
    assert first.configuration_json() == second.configuration_json()
    assert first.get_config_hash() == second.get_config_hash()


def test_processing_parameters_versions_and_stage_order_change_hashes():
    original = ConfigLink({"factor": 1})
    changed = ConfigLink({"factor": 2})
    versioned = ConfigLink({"factor": 1})
    versioned.CONFIG_VERSION = 2
    assert original.get_config_hash() != changed.get_config_hash()
    assert original.get_config_hash() != versioned.get_config_hash()

    forward = Pipeline([ConfigLink({"factor": 1}), ConfigLink({"factor": 2})])
    reverse = Pipeline([ConfigLink({"factor": 2}), ConfigLink({"factor": 1})])
    assert forward.get_config_hash() != reverse.get_config_hash()


def test_telemetry_links_do_not_change_the_processing_configuration():
    baseline = Pipeline([ConfigLink({"factor": 1}), ConfigLink({"factor": 2})])
    observed = Pipeline([
        SpeedometerLink(), ConfigLink({"factor": 1}),
        SpeedometerLink(), ConfigLink({"factor": 2}), SpeedometerLink(),
    ])
    assert observed.get_config() == baseline.get_config()
    assert observed.get_config_hash() == baseline.get_config_hash()


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("method", ["configuration_json", "get_config_json"])
def test_nonfinite_values_cannot_produce_a_configuration_hash_input(value, method):
    with pytest.raises(ValueError):
        getattr(ConfigLink({"value": value}), method)()
