# EmbeddingsCreator
# This pipeline link accepts AudioBuffers and runs them through a pre-loaded ONNX embedding model. The resulting model outputs are not saved; after inference completes, the original AudioBuffer is passed to the next link.
#
# AudioBuffers are assumed to already contain mono float audio at the sample rate and length required by the configured model.

from pathlib import Path

import numpy as np
import onnxruntime as ort

from src.AudioBuffer import AudioBuffer
from src.PipelineLink import PipelineLink
ort.preload_dlls(directory="")

class EmbeddingsCreator(PipelineLink):
    def __init__(
        self,
        model_path: str,
        use_cuda: bool = True,
    ) -> None:
        super().__init__()
        
        self.model_path = model_path
        
        model_path_object = Path(model_path)

        if not model_path_object.is_file():
            raise FileNotFoundError(
                f"ONNX model does not exist: {model_path_object}"
            )

        available_providers = ort.get_available_providers()

        if (
            use_cuda
            and "CUDAExecutionProvider" in available_providers
        ):
            providers = [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ]
        else:
            providers = [
                "CPUExecutionProvider",
            ]

        self.session = ort.InferenceSession(
            str(model_path_object),
            providers=providers,
        )

        model_inputs = self.session.get_inputs()

        if len(model_inputs) != 1:
            raise ValueError(
                "EmbeddingsCreator expects the ONNX model to have "
                f"one input, but found {len(model_inputs)}"
            )

        self.input_name = model_inputs[0].name
        self.input_shape = model_inputs[0].shape
        self.input_type = model_inputs[0].type
    
    def configuration_parameters(self) -> dict[str, any]:
        return {
            "model_path": self.model_path, #beware, changes in model path will force new embeddings! 
        }

    def next_audio(
        self,
        audio: AudioBuffer,
    ) -> None:
        if not isinstance(audio, AudioBuffer):
            raise TypeError(
                "EmbeddingsCreator expects an AudioBuffer"
            )

        model_input = self._prepare_input(
            audio
        )

        # Run all model outputs. They are deliberately discarded because this
        # link currently exists only to perform inference before forwarding
        # the source AudioBuffer.
        self.session.run(
            None,
            {
                self.input_name: model_input,
            },
        )

        if self.callback is not None:
            self.callback.next_audio(
                audio
            )

    def _prepare_input(
        self,
        audio: AudioBuffer,
    ) -> np.ndarray:
        waveform = np.asarray(
            audio.waveform,
            dtype=np.float32,
        )

        if waveform.size == 0:
            raise ValueError(
                "Cannot run inference on an empty AudioBuffer"
            )

        # Convert internal mono shape (samples, 1) to (samples,).
        if waveform.ndim == 2:
            if waveform.shape[1] != 1:
                raise ValueError(
                    "EmbeddingsCreator expects mono audio, but received "
                    f"{waveform.shape[1]} channels"
                )

            waveform = waveform[:, 0]

        elif waveform.ndim != 1:
            raise ValueError(
                "AudioBuffer waveform must have shape "
                "(samples,) or (samples, 1)"
            )

        valid_samples = min(
            audio.valid_samples,
            len(waveform),
        )

        waveform = waveform[:valid_samples]

        expected_samples = self._get_expected_samples()

        if expected_samples is not None:
            if len(waveform) < expected_samples:
                waveform = np.pad(
                    waveform,
                    (
                        0,
                        expected_samples - len(waveform),
                    ),
                )

            elif len(waveform) > expected_samples:
                raise ValueError(
                    f"AudioBuffer contains {len(waveform)} samples, "
                    f"but the model expects {expected_samples}"
                )

        # Model input shape: (batch, samples).
        return np.ascontiguousarray(
            waveform[np.newaxis, :],
            dtype=np.float32,
        )

    def _get_expected_samples(
        self,
    ) -> int | None:
        if len(self.input_shape) != 2:
            return None

        sample_dimension = self.input_shape[1]

        if isinstance(sample_dimension, int):
            return sample_dimension

        # The model has a dynamic sample dimension.
        return None

