"""ONNX Runtime provider diagnostics used by the Tools menu.

Provider enumeration is not hardware detection and does not verify that a model
can run on a reported provider.  Keep selection consistent with EmbeddingsCreator.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class ONNXRuntimeReport:
    version: str | None
    available_providers: tuple[str, ...]
    selected_providers: tuple[str, ...]
    error: str | None = None


def embedding_providers(use_cuda: bool, available_providers) -> list[str]:
    """Match the embedding stage's existing CUDA-or-CPU policy.

    CoreML and other providers can appear in ONNX Runtime but are not selected
    by Caracal's current embedding stage.
    """
    if use_cuda and "CUDAExecutionProvider" in available_providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def check_onnx_runtime() -> ONNXRuntimeReport:
    """Inspect this Python environment without loading a model or probing hardware."""
    try:
        import onnxruntime as ort  # Imported only when diagnostics are opened.

        providers = tuple(ort.get_available_providers())
        return ONNXRuntimeReport(
            version=str(getattr(ort, "__version__", "Unknown")),
            available_providers=providers,
            selected_providers=tuple(embedding_providers(True, providers)),
        )
    except Exception as exc:
        return ONNXRuntimeReport(
            version=None,
            available_providers=(),
            selected_providers=(),
            error=f"{type(exc).__name__}: {exc}",
        )


def format_report(report: ONNXRuntimeReport) -> str:
    """Plain text suitable for copying into a support request."""
    lines = ["Caracal — ONNX Runtime Diagnostics", ""]
    if report.error:
        lines.extend([
            f"ONNX Runtime: Could not load or query ({report.error})",
            "Check the ONNX Runtime installation and native-library dependencies.",
            "The diagnostic does not test GPU hardware or load the PERCH model.",
        ])
        return "\n".join(lines)

    providers = report.available_providers
    lines.extend([
        f"ONNX Runtime version: {report.version}",
        "Available execution providers:",
    ])
    lines.extend(f"  - {provider}" for provider in providers)
    if not providers:
        lines.append("  (None reported)")
    lines.extend([
        "",
        "Caracal embedding preference (with 'Use CUDA' enabled):",
        f"  {', '.join(report.selected_providers)}",
        "Caracal embedding preference (with 'Use CUDA' disabled):",
        "  CPUExecutionProvider",
        "",
    ])
    if "CUDAExecutionProvider" in providers:
        lines.append("CUDA is listed, and Caracal would request it when Use CUDA is enabled.")
    elif "CoreMLExecutionProvider" in providers:
        lines.append("CoreML is listed, but this Caracal embedding stage currently selects CPU, not CoreML.")
    else:
        lines.append("CUDA is not listed; Caracal's current embedding stage selects CPU.")
        lines.append("This does not mean the machine has no GPU.")
    lines.extend([
        "",
        "Listed providers do not prove hardware availability or that PERCH inference",
        "can run on a provider; an ONNX model session was not created.",
    ])
    return "\n".join(lines)
