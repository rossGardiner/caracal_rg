"""Regression tests for rolling-window pipeline speedometer metrics."""

from unittest.mock import Mock

import numpy as np
import pytest

from src.AudioBuffer import AudioBuffer
from src.AudioPacket import AudioPacket
from src.PipelineDefinition import PipelineDefinition, StageDefinition
from src.PipelineFactory import PipelineFactory
from src.SpeedometerLink import SpeedometerLink


def _audio(duration_s):
    sample_rate = 10
    valid_samples = round(duration_s * sample_rate)
    return AudioBuffer(
        packet=AudioPacket(recording_id="recording"),
        waveform=np.zeros(valid_samples, dtype=np.float32),
        sample_rate=sample_rate,
        start_offset_s=0.0,
        valid_samples=valid_samples,
    )


def _set_clock(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("src.SpeedometerLink.time.perf_counter", lambda: clock[0])
    return clock


def test_speedup_uses_audio_duration_in_rolling_window(monkeypatch, capsys):
    clock = _set_clock(monkeypatch)
    speed = SpeedometerLink(report_interval_s=1.0, rolling_window_s=3.0)
    sink = Mock()
    speed.register_callback(sink)

    clock[0] = 1.0
    speed.next_audio(_audio(1.0))
    assert "real-time speedup: 1.00x" in capsys.readouterr().out

    clock[0] = 2.0
    speed.next_audio(_audio(2.0))
    assert "real-time speedup: 1.50x" in capsys.readouterr().out

    # At t=4 the buffer from t=1 has left the 3-second window.
    # Only 2.0 + 0.5 seconds of audio were completed in the last 3 seconds.
    clock[0] = 4.0
    speed.next_audio(_audio(0.5))
    output = capsys.readouterr().out
    assert "rolling 3s average" in output
    assert "0.67 buffers/s" in output
    assert "real-time speedup: 0.83x" in output
    assert "total 3" in output
    assert sink.next_audio.call_count == 3


def test_idle_periods_reduce_rolling_speedup(monkeypatch, capsys):
    clock = _set_clock(monkeypatch)
    speed = SpeedometerLink(report_interval_s=1.0, rolling_window_s=3.0)

    clock[0] = 1.0
    speed.next_audio(_audio(2.0))
    assert "real-time speedup: 2.00x" in capsys.readouterr().out

    # The old audio should not inflate the new speedup after a long pause.
    clock[0] = 12.0
    speed.next_audio(_audio(1.0))
    output = capsys.readouterr().out
    assert "real-time speedup: 0.33x" in output
    assert "total 2" in output
    assert len(speed._recent_buffers) == 1


def test_early_reports_use_elapsed_time_not_full_window(monkeypatch, capsys):
    clock = _set_clock(monkeypatch)
    speed = SpeedometerLink(report_interval_s=1.0, rolling_window_s=10.0)
    clock[0] = 1.0
    speed.next_audio(_audio(3.0))
    assert "real-time speedup: 3.00x" in capsys.readouterr().out


def test_speedometer_includes_downstream_processing_time(monkeypatch, capsys):
    clock = _set_clock(monkeypatch)
    speed = SpeedometerLink(report_interval_s=1.0, rolling_window_s=10.0)
    sink = Mock()

    def expensive_downstream(_audio_buffer):
        clock[0] += 2.0

    sink.next_audio.side_effect = expensive_downstream
    speed.register_callback(sink)

    clock[0] = 1.0
    speed.next_audio(_audio(1.0))
    # One second upstream + two seconds downstream for one second of audio.
    assert "real-time speedup: 0.33x" in capsys.readouterr().out
    sink.next_audio.assert_called_once()


def test_reporting_interval_is_independent_of_rolling_window(monkeypatch, capsys):
    clock = _set_clock(monkeypatch)
    speed = SpeedometerLink(report_interval_s=2.0, rolling_window_s=10.0)
    clock[0] = 1.0
    speed.next_audio(_audio(1.0))
    assert capsys.readouterr().out == ""
    clock[0] = 2.0
    speed.next_audio(_audio(1.0))
    assert "real-time speedup: 1.00x" in capsys.readouterr().out


def test_old_stage_definitions_get_default_rolling_window():
    definition = PipelineDefinition(
        "Speedometer", "old definition",
        (StageDefinition("speedometer", "Speedometer", {"report_interval_s": 2.0}),),
    )
    speed = PipelineFactory().build(definition).head
    assert speed.report_interval_s == 2.0
    assert speed.rolling_window_s == 10.0
    assert speed.configuration_parameters() == {}


def test_rolling_window_is_editable_in_stage_definition():
    definition = PipelineDefinition(
        "Speedometer", "custom window",
        (StageDefinition(
            "speedometer", "Speedometer", {"report_interval_s": 1.0, "rolling_window_s": 5.0}
        ),),
    )
    speed = PipelineFactory().build(definition).head
    assert speed.rolling_window_s == 5.0


@pytest.mark.parametrize("parameter", ["report_interval_s", "rolling_window_s"])
def test_speedometer_rejects_non_positive_intervals(parameter):
    with pytest.raises(ValueError, match="must be positive"):
        SpeedometerLink(**{parameter: 0.0})
