"""Playback-only peak levelling and stereo routing.

All inputs receive the same automatic level treatment, regardless of source.
The listener's volume is applied *afterwards* by QAudioSink.setVolume, so it
can be changed during playback without recalculating any samples.

Never use this helper in embedding generation or model training.
"""

from dataclasses import dataclass
import math

import numpy as np


AUTO_TARGET_PEAK_DBFS = -1.0
MAX_AUTO_GAIN_DB = 48.0


@dataclass(frozen=True)
class PreparedPlayback:
    waveform: np.ndarray  # float32, contiguous, (frames, 2)
    applied_gain_db: float
    peak_limited: bool


def prepare_playback(waveform) -> PreparedPlayback:
    """Return a playback-only, automatically levelled stereo copy.

    Raise quiet peaks toward -1 dBFS, with at most +48 dB amplification.
    Already-loud recordings are not boosted. An out-of-range source is
    attenuated uniformly to prevent PCM clipping, without hard clipping
    individual samples. Silence stays silent; source samples are untouched.

    Apply the listener's volume later using QAudioSink.setVolume (0..1).
    """
    signal = np.asarray(waveform, dtype=np.float32)
    if signal.ndim == 1:
        signal = signal[:, None]
    if signal.ndim != 2 or not signal.shape[0] or signal.shape[1] not in (1, 2):
        raise ValueError("Playback requires nonempty mono or stereo audio")
    if not np.isfinite(signal).all():
        raise ValueError("Playback audio contains non-finite samples")

    peak = float(np.max(np.abs(signal)))
    gain = 1.0
    if peak > 0.0:
        target = 10.0 ** (AUTO_TARGET_PEAK_DBFS / 20.0)
        gain = min(
            10.0 ** (MAX_AUTO_GAIN_DB / 20.0),
            max(1.0, target / peak),
        )

    # Only out-of-range source values need peak protection: the calculated
    # automatic gain never targets above -1 dBFS.
    max_factor = (1.0 - 1e-5) / peak if peak else math.inf
    limited = gain > max_factor
    gain = min(gain, max_factor)

    stereo = np.repeat(signal, 2, axis=1) if signal.shape[1] == 1 else signal
    prepared = np.ascontiguousarray(stereo * gain, dtype=np.float32)
    applied_db = 20.0 * math.log10(gain) if gain > 0 else -math.inf
    return PreparedPlayback(prepared, applied_db, limited)
