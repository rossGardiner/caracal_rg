import json
from pathlib import Path

import pytest

from src.PipelineDefinition import PipelineDefinition, StageDefinition
from src.PipelineDefinitionStore import PipelineDefinitionStore


def definition(name="Example"):
    return PipelineDefinition(
        name, "A saved pipeline", (StageDefinition("gain", "Gain", {"factor": 2}),)
    )


@pytest.fixture
def store(tmp_path):
    return PipelineDefinitionStore(tmp_path / "nested" / "pipelines.json")


def test_default_path_uses_xdg_config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "config"))
    assert PipelineDefinitionStore().path == tmp_path / "config" / "caracal" / "pipelines.json"


def test_default_path_falls_back_to_the_user_config_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    assert PipelineDefinitionStore.default_path() == tmp_path / ".config" / "caracal" / "pipelines.json"


def test_missing_store_returns_no_definitions_without_creating_a_file(store):
    assert store.load() == ()
    assert not store.path.exists()


def test_save_and_load_preserve_order_version_and_editable_parameters(store):
    definitions = (definition("First"), definition("Second"))

    store.save(definitions)

    assert store.load() == definitions
    payload = json.loads(store.path.read_text())
    assert payload == {"version": 1, "pipelines": [item.to_dict() for item in definitions]}
    assert store.path.read_text().endswith("\n")
    assert list(store.path.parent.iterdir()) == [store.path]


def test_empty_save_replaces_existing_definitions(store):
    store.save((definition(),))
    store.save(())
    assert store.load() == ()
    assert json.loads(store.path.read_text()) == {"version": 1, "pipelines": []}


@pytest.mark.parametrize(
    "payload",
    [[], {"pipelines": []}, {"version": 0}, {"version": 2}, {"version": 1, "pipelines": {}}],
)
def test_load_rejects_bad_payload_shapes_and_versions(store, payload):
    store.path.parent.mkdir(parents=True)
    store.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        store.load()


def test_load_rejects_malformed_json(store):
    store.path.parent.mkdir(parents=True)
    store.path.write_text("{broken JSON")
    with pytest.raises(json.JSONDecodeError):
        store.load()


def test_version_one_file_may_omit_the_optional_pipeline_list(store):
    store.path.parent.mkdir(parents=True)
    store.path.write_text('{"version":1}')
    assert store.load() == ()


def test_load_rejects_duplicate_definition_names(store):
    store.path.parent.mkdir(parents=True)
    payload = {"version": 1, "pipelines": [definition().to_dict(), definition().to_dict()]}
    store.path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unique"):
        store.load()


def test_save_rejects_duplicate_names_before_touching_the_existing_file(store):
    store.save((definition("Original"),))
    previous = store.path.read_bytes()

    with pytest.raises(ValueError, match="unique"):
        store.save((definition(), definition()))

    assert store.path.read_bytes() == previous
    assert list(store.path.parent.iterdir()) == [store.path]


@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["serialization", "write", "replace"])
def test_failed_save_preserves_existing_json_and_removes_partial_files(
    store, monkeypatch, existing, failure
):
    previous = None
    if existing:
        store.save((definition("Original"),))
        previous = store.path.read_bytes()
    candidate = definition("Replacement")
    error = OSError
    if failure == "serialization":
        candidate.stages[0].parameters["not-json"] = object()
        error = TypeError
    elif failure == "write":
        def fail_dump(payload, handle, **kwargs):
            handle.write("partial JSON")
            raise OSError("Injected write failure")
        monkeypatch.setattr("src.PipelineDefinitionStore.json.dump", fail_dump)
    else:
        real_replace = Path.replace

        def fail_replace(source, destination):
            if Path(destination) == store.path:
                assert Path(source).is_file()
                raise OSError("Injected replace failure")
            return real_replace(source, destination)

        monkeypatch.setattr(Path, "replace", fail_replace)

    with pytest.raises(error):
        store.save((candidate,))

    assert list(store.path.parent.iterdir()) == ([store.path] if existing else [])
    if existing:
        assert store.path.read_bytes() == previous
        assert store.load()[0].name == "Original"
    else:
        assert not store.path.exists()
