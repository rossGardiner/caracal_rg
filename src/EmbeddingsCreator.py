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
    """Run an ONNX embedding model on AudioBuffer objects and attach results.

    Loads an ONNX model via onnxruntime and performs inference on incoming
    AudioBuffers, storing outputs under audio.embeddings[embedding_name].
    """
    def __init__(
        self,
        model_path,
        use_cuda=True,
        embedding_name="perch_v2",
    ):
        """Initialize the embeddings runtime.

        Args:
            model_path (str): Path to the ONNX model file; must exist.
            use_cuda (bool): Prefer CUDAExecutionProvider if available.
            embedding_name (str): Key name under which outputs are stored on
                audio.embeddings.

        Raises:
            FileNotFoundError: If model_path does not exist.
            ValueError: If the model has an unexpected number of inputs.
        """
        super().__init__()

        self.embedding_name = embedding_name
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
        """Return configuration parameters affecting the embedding output.

        Returns:
            dict: Includes 'model_path' (note: changing the path forces different pipeline hash).
        """
        return {
            "model_path": self.model_path, #beware, changes in model path will force new embeddings!
        }
    
    def next_audio(self, audio):
        """Run the ONNX model to compute embeddings and attach them to audio.

        If the embedding is already present, the buffer is forwarded unchanged.

        Args:
            audio (AudioBuffer): Buffer of mono audio at the expected sample rate.

        Raises:
            TypeError: If audio is not an AudioBuffer.
            ValueError: For empty waveforms or channel count mismatches.
            RuntimeError: If the ONNX model returns no outputs.
        """
        if not isinstance(audio, AudioBuffer):
            raise TypeError(
                "EmbeddingsCreator expects an AudioBuffer"
            )
        if self.embedding_name in audio.embeddings:
            if self.callback is not None:
                self.callback.next_audio(audio)
            return
        print("computing")
        model_input = self._prepare_input(audio)

        outputs = self.session.run(
            None,
            {
                self.input_name: model_input,
            },
        )

        if len(outputs) == 0:
            raise RuntimeError(
                "The ONNX model returned no outputs"
            )

        pipeline_hash = self.get_config_hash()

        audio.embeddings[self.embedding_name] = {
            "values": np.asarray(outputs[0]),
            "pipeline_hash": pipeline_hash,
            "metadata": {
                "input_name": self.input_name,
                "input_shape": self.input_shape,
                "output_index": 0,
            },
            "loaded_from_cache": False,
        }

        if self.callback is not None:
            self.callback.next_audio(audio)

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

