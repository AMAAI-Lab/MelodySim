import numpy as np
import pytest

from data.augmentation.audio import prepare_segments, segment_starts, source_intervals
from data.augmentation.config import AudioConfig


def test_short_and_single_segment_keep_batch_dimension():
    config = AudioConfig(sample_rate=8000, window_seconds=1, hop_seconds=1)
    for length in (1200, 8000):
        segments = list(prepare_segments(np.ones(length, dtype=np.float32) * 0.2, config))
        assert len(segments) == 1 and segments[0][0].shape == (8000,)
        assert segments[0][1]["valid_input_samples"] == length


def test_circular_shift_source_mapping():
    assert source_intervals(0, 4, 10, 2, 1) == [[8.0, 10.0], [0.0, 2.0]]
    assert source_intervals(4, 4, 10, -2, 1) == [[6.0, 10.0]]


@pytest.mark.parametrize("mode", ["speed", "stretch"])
def test_pitch_and_tempo_have_documented_frequency(mode):
    sr = 8000
    tone = np.sin(2 * np.pi * 440 * np.arange(sr * 2) / sr).astype(np.float32) * 0.2
    config = AudioConfig(sample_rate=sr, window_seconds=2, hop_seconds=2)
    plan = {"pitch_shift_semitones": 2, "time_shift_samples": 0, "tempo_factor": 1.1, "tempo_mode": mode}
    output, _ = next(prepare_segments(tone, config, plan))
    assert abs(len(output) - round(len(tone) / 1.1)) <= 1
    spectrum = abs(np.fft.rfft(output[1000:-1000]))
    pitch = np.fft.rfftfreq(len(output) - 2000, 1 / sr)[spectrum.argmax()]
    expected = 440 * 2 ** (2 / 12) * (1.1 if mode == "speed" else 1)
    assert abs(pitch - expected) < 3


def test_padding_and_canonical_length_align_versions():
    config = AudioConfig(sample_rate=100, window_seconds=1, hop_seconds=1, tail_policy="pad")
    plan = {"pitch_shift_semitones": 0, "time_shift_samples": 0, "tempo_factor": 1, "tempo_mode": "speed"}
    segments = list(prepare_segments(np.ones(240, dtype=np.float32), config, plan, 250))
    assert len(segments) == 3
    assert segments[-1][1]["padded_input_samples"] == 50
    assert max(abs(segments[0][0])) <= 0.98


def test_overlapping_windows_keep_all_partial_tail_starts():
    assert segment_starts(250, 100, 25, "pad") == list(range(0, 250, 25))
    assert segment_starts(250, 100, 25, "drop") == list(range(0, 151, 25))


def test_default_tempo_mode_preserves_the_sampled_integer_transposition():
    assert AudioConfig().tempo_mode == "stretch"


def test_pitch_shift_is_continuous_across_clip_boundaries():
    import librosa
    sr = 8000
    wave = 0.15 * np.sin(2 * np.pi * 440 * np.arange(sr * 3) / sr).astype(np.float32)
    config = AudioConfig(sample_rate=sr, window_seconds=1, hop_seconds=1)
    plan = {"pitch_shift_semitones": 3, "time_shift_samples": 200, "tempo_factor": 1.0, "tempo_mode": "stretch"}
    actual = np.concatenate([clip for clip, _ in prepare_segments(wave, config, plan)])
    whole_shifted = librosa.effects.pitch_shift(wave, sr=sr, n_steps=3)
    np.testing.assert_allclose(actual, np.roll(whole_shifted, 200), atol=1e-6)
