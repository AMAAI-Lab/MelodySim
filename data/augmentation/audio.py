"""Audio operations and exact source-time mappings for positive pairs."""

from fractions import Fraction
import math

import numpy as np
from scipy.signal import resample_poly
import soundfile as sf


def read_audio(path, sample_rate):
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    waveform = waveform.mean(axis=1)
    if source_rate != sample_rate:
        ratio = Fraction(sample_rate, source_rate)
        waveform = resample_poly(waveform, ratio.numerator, ratio.denominator).astype(np.float32)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"Audio is empty or contains nonfinite samples: {path}")
    return waveform, {"source_sample_rate": source_rate, "output_sample_rate": sample_rate}


def sample_audio_plan(rng, config):
    shift = round(config.time_shift_seconds * config.sample_rate)
    return {"pitch_shift_semitones": int(rng.integers(-config.pitch_shift_semitones, config.pitch_shift_semitones + 1)),
            "time_shift_samples": int(rng.integers(-shift, shift + 1)),
            "time_shift_mode": "circular", "tempo_factor": float(rng.uniform(*config.tempo_range)),
            "tempo_mode": config.tempo_mode}


def segment_starts(length, window, hop, tail_policy):
    if length <= window:
        return [0]
    return list(range(0, length if tail_policy == "pad" else length - window + 1, hop))


def source_intervals(start, valid_samples, length, shift, sample_rate):
    position = (start - shift) % length
    first = min(valid_samples, length - position)
    intervals = [[position / sample_rate, (position + first) / sample_rate]] if first else []
    if first < valid_samples:
        intervals.append([0.0, (valid_samples - first) / sample_rate])
    return intervals


def prepare_segments(waveform, config, plan=None, canonical_samples=None):
    canonical_samples = len(waveform) if canonical_samples is None else canonical_samples
    if canonical_samples < 1:
        raise ValueError("Canonical audio length must be positive")
    data = np.zeros(canonical_samples, dtype=np.float32)
    data[:min(len(waveform), canonical_samples)] = waveform[:canonical_samples]
    # Figure 1: pitch shift the whole waveform before time shift/segmentation.
    # Per-clip pitch shifting creates a separate phase-vocoder edge at every cut.
    if plan and plan["pitch_shift_semitones"]:
        import librosa
        data = librosa.effects.pitch_shift(data, sr=config.sample_rate, n_steps=plan["pitch_shift_semitones"])
    shift = plan["time_shift_samples"] if plan else 0
    data = np.roll(data, shift)
    window, hop = round(config.window_seconds * config.sample_rate), round(config.hop_seconds * config.sample_rate)
    for index, start in enumerate(segment_starts(canonical_samples, window, hop, config.tail_policy)):
        valid = min(window, canonical_samples - start)
        segment = np.pad(data[start:start + valid], (0, window - valid))
        if plan:
            factor = plan["tempo_factor"]
            if factor != 1:
                if plan["tempo_mode"] == "stretch":
                    import librosa
                    segment = librosa.effects.time_stretch(segment, rate=factor)
                elif plan["tempo_mode"] == "speed":
                    import librosa
                    segment = librosa.resample(segment, orig_sr=config.sample_rate * factor, target_sr=config.sample_rate, res_type="soxr_hq")
                else:
                    raise ValueError(f"Unknown tempo mode: {plan['tempo_mode']}")
        peak = float(np.max(np.abs(segment)))
        scale = 0.98 / peak if peak > 0.98 else 1.0
        segment = np.asarray(segment * scale, dtype=np.float32)
        if not len(segment) or not np.isfinite(segment).all():
            raise ValueError("Augmentation produced invalid audio")
        yield segment, {"index": index, "source_window_start_seconds": start / config.sample_rate,
                        "source_intervals_seconds": source_intervals(start, valid, canonical_samples, shift, config.sample_rate),
                        "valid_input_samples": valid, "padded_input_samples": window - valid,
                        "output_samples": len(segment), "peak_scale": scale,
                        "output_to_input_time_scale": plan["tempo_factor"] if plan else 1.0,
                        "speed_pitch_offset_semitones": 12 * math.log2(plan["tempo_factor"]) if plan and plan["tempo_mode"] == "speed" else 0.0}
