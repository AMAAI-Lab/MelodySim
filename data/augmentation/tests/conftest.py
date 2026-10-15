from pathlib import Path

import pretty_midi
import pytest

from data.augmentation.config import GenerationConfig
from data.augmentation.tools.create_augmentation_demo import make_midi


@pytest.fixture
def midi(tmp_path):
    return make_midi(tmp_path / "fixture.mid", seconds=4.0)


@pytest.fixture
def config(tmp_path):
    source = tmp_path / "source"
    make_midi(source / "train" / "TrackA" / "all_src.mid", seconds=4.0)
    make_midi(source / "train" / "TrackB" / "all_src.mid", offset=2, seconds=4.0)
    cfg = GenerationConfig(input_root=str(source), output_root=str(tmp_path / "generated"),
                           soundfont=str(Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"),
                           original_mode="synthesize", num_versions=2)
    cfg.audio.sample_rate = 8000
    cfg.audio.window_seconds = 2.0
    cfg.audio.hop_seconds = 2.0
    cfg.audio.pitch_shift_semitones = 1
    cfg.audio.time_shift_seconds = 0.25
    return cfg.validate()
