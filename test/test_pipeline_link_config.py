import json
import hashlib
from src.PipelineLink import PipelineLink

class TestLink(PipelineLink):
    def __init__(self, params):
        super().__init__()
        self._params = params

    def configuration_parameters(self) -> dict[str, any]:
        return self._params


def test_configuration_serialization_and_hash():
    link = TestLink({"alpha": 1, "beta": "x"})
    cfg = link.configuration()
    assert cfg["link_type"] == "TestLink"
    assert "version" in cfg and "parameters" in cfg

    cfg_json = link.configuration_json()
    assert json.loads(cfg_json) == cfg

    expected_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
    assert link.configuration_hash() == expected_hash


def test_get_config_links_respects_chain_order():
    a = TestLink({"a": 1})
    b = TestLink({"b": 2})
    c = TestLink({"c": 3})

    a.register_callback(b)
    b.register_callback(c)

    links = c.get_config_links()
    assert [l["link_type"] for l in links] == ["TestLink", "TestLink", "TestLink"]
    assert [l["parameters"] for l in links] == [{"a":1}, {"b":2}, {"c":3}]

    # Also verify json and hash on the aggregated config
    cfg = c.get_config()
    cfg_json = c.get_config_json()
    assert json.loads(cfg_json) == cfg
    expected_agg_hash = hashlib.sha256(cfg_json.encode("utf-8")).hexdigest()
    assert c.get_config_hash() == expected_agg_hash
