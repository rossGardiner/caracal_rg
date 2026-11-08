"""Build, smoke-test and archive Audio Analysis with PyInstaller.

Executed by .github/workflows/build-binaries.yml on the target OS. The
application is intentionally built in onedir mode to keep Qt, PyTorch and
ONNX Runtime native libraries next to the executable.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import zipfile


ROOT = Path(__file__).resolve().parents[1]
NAME = "AudioAnalysis"


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


def bundle(platform: str) -> Path:
    # No requirements or user datasets/ONNX models are silently included:
    # projects reference those resources by path. See the build README.
    stylesheet = ROOT / "src" / "styles" / "application.qss"
    if not stylesheet.is_file():
        raise SystemExit(f"Missing stylesheet resource: {stylesheet}")
    args = [
        sys.executable, "-m", "PyInstaller",
        "--noconfirm", "--clean", "--onedir", "--windowed",
        "--name", NAME,
        "--paths", str(ROOT),
        "--add-data", f"{stylesheet}{os.pathsep}src/styles",
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
    executable = bundle(args.platform)
    smoke_test(executable)
    create_archive(args.platform)


if __name__ == "__main__":
    main()
