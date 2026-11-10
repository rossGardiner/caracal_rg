"""Build, smoke-test and archive Audio Analysis with PyInstaller.

Executed by .github/workflows/build-binaries.yml on the target OS. The
application is intentionally built in onedir mode to keep Qt, PyTorch and
ONNX Runtime native libraries next to the executable.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import tempfile
import urllib.request
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NAME = "Caracal_RG"


def run(command: list[str], *, env: dict[str, str] | None = None) -> None:
    print("Running:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def ensure_parser_available() -> None:
    try:
        from caracal import SyslogParser  # noqa: F401
    except (ImportError, ModuleNotFoundError) as exc:
        raise SystemExit(
            "Cannot build the Caracal adapter: `from caracal import SyslogParser` "
            "failed. Install the parser package in GitHub Actions first. "
            "See BUILDING_BINARIES.md."
        ) from exc


def ensure_model_available() -> Path:
    """Get the Perch model and verify its SHA-256 before bundling.

    GitHub Actions must set PERCH_ONNX_URL and PERCH_ONNX_SHA256 as env vars.
    Developers can build offline when they already have a local model file.
    """
    model = ROOT / "assets" / "perch_v2.onnx"
    url = os.environ.get("PERCH_ONNX_URL", "").strip()
    checksum = os.environ.get("PERCH_ONNX_SHA256", "").strip().lower()

    if checksum and (len(checksum) != 64 or any(c not in "0123456789abcdef" for c in checksum)):
        raise SystemExit("PERCH_ONNX_SHA256 must contain 64 hexadecimal characters")
    if os.environ.get("CI") and not checksum:
        raise SystemExit("PERCH_ONNX_SHA256 must be set for CI builds")

    def digest(path: Path) -> str:
        hasher = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    if model.is_file() and model.stat().st_size > 0:
        if not checksum or digest(model) == checksum:
            print(f"Using Perch model: {model}", flush=True)
            return model
        if not url:
            raise SystemExit("Local Perch model checksum mismatch; set PERCH_ONNX_URL to redownload")
        print("Local Perch model checksum mismatch; downloading verified copy", flush=True)

    if not url:
        raise SystemExit(
            "Perch ONNX model is missing at assets/perch_v2.onnx. "
            "Set PERCH_ONNX_URL and PERCH_ONNX_SHA256 or provide a verified model locally."
        )
    if not url.startswith("https://"):
        raise SystemExit("PERCH_ONNX_URL must be an HTTPS URL")
    if not checksum:
        raise SystemExit("PERCH_ONNX_SHA256 is required when downloading the model")

    model.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(dir=model.parent, prefix=".perch-", suffix=".part", delete=False) as tmp:
            tmp_path = Path(tmp.name)
            req = urllib.request.Request(url, headers={"User-Agent": "AudioAnalysis-CI/1.0"})
            with urllib.request.urlopen(req, timeout=120) as response:
                for chunk in iter(lambda: response.read(1024 * 1024), b""):
                    tmp.write(chunk)
        actual = digest(tmp_path)
        if actual != checksum:
            raise SystemExit(f"Perch model SHA-256 mismatch: expected {checksum}, got {actual}")
        if tmp_path.stat().st_size == 0:
            raise SystemExit("Downloaded Perch model is empty")
        os.replace(tmp_path, model)
        print(f"Downloaded and verified Perch ONNX model: {model}", flush=True)
        return model
    finally:
        if tmp_path is not None:
            tmp_path.unlink(missing_ok=True)


def bundle(platform: str, model: Path) -> Path:
    # User recordings and embedding caches are referenced by path, not bundled.
    # During the application rename this resource was called caracal.qss.
    # Accept either checkout layout, and bundle both names so either version
    # of ApplicationTheme.py can find the stylesheet at runtime.
    styles_dir = ROOT / "src" / "styles"
    stylesheet = next((p for p in (
        styles_dir / "application.qss", styles_dir / "caracal.qss"
    ) if p.is_file()), None)
    if stylesheet is None:
        raise SystemExit(
            "Missing stylesheet: expected src/styles/application.qss or "
            "src/styles/caracal.qss. Check that the stylesheet is tracked by git."
        )

    staged_styles = ROOT / "build" / "ci-styles"
    staged_styles.mkdir(parents=True, exist_ok=True)
    for filename in ("application.qss", "caracal.qss"):
        shutil.copyfile(stylesheet, staged_styles / filename)
    print(f"Bundling stylesheet from {stylesheet.relative_to(ROOT)}", flush=True)

    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onedir", "--windowed",
        "--name", NAME,
        "--paths", str(ROOT),
        "--add-data", f"{model}{os.pathsep}assets",
        "--add-data", f"{staged_styles / 'application.qss'}{os.pathsep}src/styles",
        "--add-data", f"{staged_styles / 'caracal.qss'}{os.pathsep}src/styles",
        # PipelineFactory loads the Caracal stage dynamically. Make sure it
        # and the parser integration are included, not just the entrypoint.
        "--hidden-import", "src.CaracalStreamer",
        "--collect-all", "caracal",
        "--collect-submodules", "onnxruntime",
        str(ROOT / "src" / "main.py"),
    ]
    run(args)

    if platform.startswith("macos"):
        expected = ROOT / "dist" / f"{NAME}.app" / "Contents" / "MacOS" / NAME
    else:
        suffix = ".exe" if platform.startswith("windows") else ""
        expected = ROOT / "dist" / NAME / f"{NAME}{suffix}"
    if not expected.is_file():
        raise SystemExit(f"PyInstaller did not create the expected executable: {expected}")
    return expected


def smoke_test(executable: Path) -> None:
    # Startup-only smoke test. The application exits with code 0 after the
    # Getting Started window has been created and the event loop has started.
    env = os.environ.copy()
    env["AUDIO_ANALYSIS_CI_SMOKE_TEST"] = "1"
    env["QT_QPA_PLATFORM"] = "offscreen"
    run([str(executable), "-platform", "offscreen"], env=env)


def create_archive(platform: str) -> Path:
    release = ROOT / "release"
    release.mkdir(exist_ok=True)

    if platform.startswith("macos"):
        app = ROOT / "dist" / f"{NAME}.app"
        output = release / f"{NAME}-{platform}.zip"
        # ditto preserves symlinks, permissions and bundle metadata. The
        # ordinary Python zipfile writer is NOT sufficient for a macOS .app.
        run(["ditto", "-c", "-k", "--sequesterRsrc", "--keepParent",
             str(app), str(output)])
    elif platform.startswith("windows"):
        directory = ROOT / "dist" / NAME
        output = release / f"{NAME}-{platform}.zip"
        with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED,
                             compresslevel=6, allowZip64=True) as archive:
            for source in sorted(directory.rglob("*")):
                if source.is_file():
                    archive.write(source, f"{NAME}/{source.relative_to(directory).as_posix()}")
    else:
        directory = ROOT / "dist" / NAME
        output = release / f"{NAME}-{platform}.tar.gz"
        with tarfile.open(output, "w:gz", format=tarfile.PAX_FORMAT) as archive:
            archive.add(directory, arcname=NAME)

    if not output.is_file() or output.stat().st_size == 0:
        raise SystemExit(f"Packaging failed: {output}")
    print(f"Created: {output} ({output.stat().st_size / (1024**2):.1f} MiB)")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--platform", required=True, choices=(
        "windows-x64", "linux-x64", "macos-arm64", "macos-x64"
    ))
    args = parser.parse_args()
    ensure_parser_available()
    model = ensure_model_available()
    executable = bundle(args.platform, model)
    smoke_test(executable)
    create_archive(args.platform)


if __name__ == "__main__":
    main()

