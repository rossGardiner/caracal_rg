"""Bundled asset paths must not depend on the process working directory."""

import importlib
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

from src.ResourcePaths import resource_path


def test_resource_path_in_checkout(monkeypatch, tmp_path):
    monkeypatch.delattr(sys, "_MEIPASS", raising=False)
    monkeypatch.chdir(tmp_path)
    repository = Path(__file__).resolve().parents[1]
    assert resource_path("assets/dummy_data") == repository / "assets" / "dummy_data"


def test_resource_path_in_frozen_bundle(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    monkeypatch.chdir(tmp_path.parent)
    assert resource_path("assets/perch_v2.onnx") == tmp_path / "assets" / "perch_v2.onnx"


def test_resource_path_resolves_macos_resource_symlinks(monkeypatch, tmp_path):
    # PyInstaller 6 places .app data in Resources and links from Frameworks.
    resources = tmp_path / "Contents" / "Resources"
    frameworks = tmp_path / "Contents" / "Frameworks"
    (resources / "assets").mkdir(parents=True)
    frameworks.mkdir()
    (resources / "assets" / "perch_v2.onnx").touch()
    try:
        (frameworks / "assets").symlink_to(resources / "assets", target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("Cannot create directory symlinks on this platform")
    monkeypatch.setattr(sys, "_MEIPASS", str(frameworks), raising=False)
    assert resource_path("assets/perch_v2.onnx") == resources / "assets" / "perch_v2.onnx"


def test_caracal_builder_only_resolves_example_data(monkeypatch, tmp_path):
    class FakeCaracalStreamer:
        def __init__(self, rootpath):
            self.rootpath = rootpath

    fake_module = ModuleType("src.CaracalStreamer")
    fake_module.CaracalStreamer = FakeCaracalStreamer
    monkeypatch.setitem(sys.modules, "src.CaracalStreamer", fake_module)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    from src.PipelineFactory import _build_caracal_streamer

    for alias in ("assets/dummy_data", r"assets\dummy_data"):
        assert _build_caracal_streamer({"rootpath": alias}, None).rootpath == str(
            tmp_path / "assets" / "dummy_data"
        )
    for custom in ("D:/customer/caracal", "relative/recordings"):
        assert _build_caracal_streamer({"rootpath": custom}, None).rootpath == custom


def test_embeddings_creator_resolves_only_bundled_default_model(monkeypatch, tmp_path):
    # Stub the optional ONNX runtime, so this test does not need the 390 MB model.
    class FakeSession:
        def get_providers(self):
            return ["CPUExecutionProvider"]

        def get_inputs(self):
            return [SimpleNamespace(name="waveform", shape=(None, 160000), type="tensor(float)")]

    opened_paths = []
    fake_ort = ModuleType("onnxruntime")
    fake_ort.preload_dlls = lambda directory: None
    fake_ort.get_available_providers = lambda: ["CPUExecutionProvider"]
    def make_session(path, providers):
        opened_paths.append(path)
        return FakeSession()

    fake_ort.InferenceSession = make_session

    old_module = sys.modules.pop("src.EmbeddingsCreator", None)
    monkeypatch.setitem(sys.modules, "onnxruntime", fake_ort)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    model = tmp_path / "assets" / "perch_v2.onnx"
    model.parent.mkdir(parents=True)
    model.touch()
    custom_model = tmp_path / "my_model.onnx"
    custom_model.touch()

    try:
        creator_module = importlib.import_module("src.EmbeddingsCreator")
        creator = creator_module.EmbeddingsCreator("assets/perch_v2.onnx", use_cuda=False)
        assert opened_paths[-1] == str(model)
        assert creator.configuration_parameters() == {"model_path": "assets/perch_v2.onnx"}

        creator_module.EmbeddingsCreator(str(custom_model), use_cuda=False)
        assert opened_paths[-1] == str(custom_model)

        with pytest.raises(FileNotFoundError, match="missing_model.onnx"):
            creator_module.EmbeddingsCreator("missing_model.onnx", use_cuda=False)
    finally:
        # Avoid retaining a fake onnxruntime dependency for later tests.
        sys.modules.pop("src.EmbeddingsCreator", None)
        if old_module is not None:
            sys.modules["src.EmbeddingsCreator"] = old_module
