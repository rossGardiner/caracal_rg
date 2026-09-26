from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SpectrogramResult:
    """
    Pure data needed to render one spectrogram.

    This object contains no Qt widgets and can therefore be created
    safely on a background worker thread.
    """

    image: np.ndarray
    duration_s: float
    max_frequency_hz: float


def calculate_spectrogram(
    waveform,
    sample_rate: int,
    n_fft: int = 1024,
    hop_length: int = 256,
):
    """
    Convert an audio waveform into display-ready spectrogram data.

    Expensive NumPy work happens here rather than inside the Qt
    Spectrogram widget. The returned SpectrogramResult can later be
    rendered cheaply on the GUI thread.
    """

    if sample_rate <= 0:
        raise ValueError(
            "sample_rate must be greater than zero"
        )

    if n_fft <= 0:
        raise ValueError(
            "n_fft must be greater than zero"
        )

    if hop_length <= 0:
        raise ValueError(
            "hop_length must be greater than zero"
        )

    waveform = np.asarray(
        waveform,
        dtype=np.float32,
    )

    if waveform.ndim == 1:
        samples = waveform
    elif waveform.ndim == 2:
        samples = waveform.mean(
            axis=1
        )
    else:
        raise ValueError(
            "waveform must be one- or two-dimensional"
        )

    if len(samples) < n_fft:
        samples = np.pad(
            samples,
            (
                0,
                n_fft - len(samples),
            ),
        )

    window = np.hanning(
        n_fft
    ).astype(
        np.float32
    )

    frames = (
        np.lib.stride_tricks
        .sliding_window_view(
            samples,
            n_fft,
        )[::hop_length]
    )

    frames = (
        frames
        * window
    )

    spectrum = np.fft.rfft(
        frames,
        axis=1,
    )

    power = (
        np.abs(
            spectrum
        ) ** 2
    )

    power_db = (
        10
        * np.log10(
            power
            + 1e-12
        )
    )

    # STFT is time x frequency. ImageItem is configured with
    # row-major ordering, so expose frequency x time.
    image = np.asarray(
        power_db.T,
        dtype=np.float32,
    )

    duration_s = (
        len(waveform)
        / sample_rate
    )

    max_frequency_hz = (
        sample_rate / 2
    )

    return SpectrogramResult(
        image=image,
        duration_s=duration_s,
        max_frequency_hz=max_frequency_hz,
    )
