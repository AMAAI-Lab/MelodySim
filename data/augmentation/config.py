"""Validated configuration; relative paths are relative to the YAML file."""

from dataclasses import asdict, dataclass, field, fields, is_dataclass
import math
import os
from pathlib import Path

import yaml


ASSET_DIR = Path(__file__).parent / "assets"


@dataclass
class MidiConfig:
    keep_instrument_probability: float = 0.2
    same_family_probability: float = 0.7
    remove_track_probability: tuple = (0.1, 0.5)
    remove_drums_probability: float = 0.5
    note_split_probability: tuple = (0.3, 0.85)
    chord_inversion_probability: tuple = (0.3, 0.85)
    arpeggiation_probability: tuple = (0.3, 0.85)
    onset_tolerance_seconds: float = 0.05
    protect_families: tuple = ("piano", "guitar", "bass")
    protect_vocals: bool = True
    split_melody_notes: bool = True


@dataclass
class AudioConfig:
    sample_rate: int = 44100
    window_seconds: float = 10.0
    hop_seconds: float = 10.0
    pitch_shift_semitones: int = 4
    time_shift_seconds: float = 3.0
    tempo_range: tuple = (0.9, 1.1)
    tempo_mode: str = "stretch"
    tail_policy: str = "drop"


@dataclass
class SynthesisConfig:
    gain: float = 0.2
    tail_seconds: float = 0.5


@dataclass
class GenerationConfig:
    input_root: str = ""
    output_root: str = ""
    soundfont: str = ""
    classifier_path: str = str(ASSET_DIR / "classifier_model.pkl")
    melody_overrides: str | None = None
    split: str | None = None
    splits: tuple = ("train", "validation", "test")
    midi_filename: str = "all_src.mid"
    audio_filename: str = "mix.flac"
    original_mode: str = "audio_mix"
    num_versions: int = 3
    seed: int = 2026
    midi: MidiConfig = field(default_factory=MidiConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    synthesis: SynthesisConfig = field(default_factory=SynthesisConfig)

    def validate(self):
        for key in ("input_root", "output_root", "soundfont", "classifier_path"):
            if not getattr(self, key):
                raise ValueError(f"{key} is required")
        for key in ("num_versions", "seed", "audio.sample_rate", "audio.pitch_shift_semitones"):
            value = self
            for part in key.split("."):
                value = getattr(value, part)
            if type(value) is not int or value < (1 if key in ("num_versions", "audio.sample_rate") else 0):
                raise ValueError(f"{key} must be a valid nonnegative integer (count/rate must be positive)")
        for name in ("keep_instrument_probability", "same_family_probability", "remove_drums_probability"):
            value = getattr(self.midi, name)
            if not isinstance(value, (int, float)) or not 0 <= value <= 1:
                raise ValueError(f"midi.{name} must be in [0, 1]")
        for name in ("remove_track_probability", "note_split_probability", "chord_inversion_probability", "arpeggiation_probability"):
            _range(getattr(self.midi, name), f"midi.{name}", 0, 1)
        _range(self.audio.tempo_range, "audio.tempo_range", 0, math.inf, positive=True)
        for name, value, allow_zero in (
            ("audio.window_seconds", self.audio.window_seconds, False),
            ("audio.hop_seconds", self.audio.hop_seconds, False),
            ("audio.time_shift_seconds", self.audio.time_shift_seconds, True),
            ("midi.onset_tolerance_seconds", self.midi.onset_tolerance_seconds, True),
            ("synthesis.gain", self.synthesis.gain, False),
            ("synthesis.tail_seconds", self.synthesis.tail_seconds, True),
        ):
            if not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (value == 0 and not allow_zero):
                raise ValueError(f"{name} has an invalid value")
        if round(self.audio.window_seconds * self.audio.sample_rate) < 1 or round(self.audio.hop_seconds * self.audio.sample_rate) < 1:
            raise ValueError("window and hop must each contain at least one sample")
        if self.original_mode not in ("audio_mix", "synthesize"):
            raise ValueError("original_mode must be audio_mix or synthesize")
        if self.audio.tempo_mode not in ("speed", "stretch"):
            raise ValueError("audio.tempo_mode must be speed or stretch")
        if self.audio.tail_policy not in ("drop", "pad"):
            raise ValueError("audio.tail_policy must be drop or pad")
        if not isinstance(self.splits, (tuple, list)) or not self.splits or len(set(self.splits)) != len(self.splits):
            raise ValueError("splits must be nonempty and unique")
        if self.split is not None and self.split not in self.splits:
            raise ValueError("split must be one of splits")
        for name in (*self.splits, self.midi_filename, self.audio_filename):
            if not isinstance(name, str) or not name or Path(name).name != name or name in (".", "..") or "/" in name or "\\" in name:
                raise ValueError("split names and input filenames must be simple path components")
        from .midi import FAMILY_NAMES
        if not isinstance(self.midi.protect_families, (tuple, list)) or not set(self.midi.protect_families) <= set(FAMILY_NAMES):
            raise ValueError(f"Unknown protected instrument family; use {FAMILY_NAMES}")
        for name in ("protect_vocals", "split_melody_notes"):
            if type(getattr(self.midi, name)) is not bool:
                raise ValueError(f"midi.{name} must be a boolean")
        source, output = Path(self.input_root).resolve(), Path(self.output_root).resolve()
        if source == output or output in source.parents or source in output.parents:
            raise ValueError("input_root and output_root must be separate, non-nested directories")
        return self

    def to_dict(self):
        return asdict(self)


def _range(value, name, low, high, positive=False):
    if not isinstance(value, (tuple, list)) or len(value) != 2:
        raise ValueError(f"{name} must be [minimum, maximum]")
    a, b = value
    if not all(isinstance(x, (int, float)) and math.isfinite(x) for x in value) or not low <= a <= b <= high or (positive and a <= 0):
        raise ValueError(f"{name} is outside its allowed range")


def _update(instance, values):
    if not isinstance(values, dict):
        raise ValueError("Configuration sections must be mappings")
    known = {f.name for f in fields(instance)}
    if set(values) - known:
        raise ValueError(f"Unknown configuration keys: {sorted(set(values) - known)}")
    for name, value in values.items():
        current = getattr(instance, name)
        if is_dataclass(current):
            _update(current, value)
        else:
            setattr(instance, name, value)


def load_config(path, **overrides):
    path = Path(path).resolve()
    config = GenerationConfig()
    with path.open(encoding="utf-8") as stream:
        _update(config, yaml.safe_load(stream) or {})
    for name, value in overrides.items():
        if value is not None:
            setattr(config, name, value)
    for name in ("input_root", "output_root", "soundfont", "classifier_path", "melody_overrides"):
        value = getattr(config, name)
        if value:
            resolved = Path(os.path.expandvars(os.path.expanduser(value)))
            setattr(config, name, str((path.parent / resolved).resolve()))
    return config.validate()
