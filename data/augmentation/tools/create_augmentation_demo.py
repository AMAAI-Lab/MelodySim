"""Create original toy compositions and use pretty_midi's bundled test SoundFont.

This is a functional demonstration, not a substitute for Slakh or Musyng.
Run: python -m data.augmentation.tools.create_augmentation_demo --run
"""

import argparse
from pathlib import Path

import pretty_midi
import yaml


def make_midi(path, offset=0, seconds=21.0):
    midi = pretty_midi.PrettyMIDI(initial_tempo=120)
    melody = pretty_midi.Instrument(73, name="Flute melody")
    bass = pretty_midi.Instrument(32, name="Bass")
    piano = pretty_midi.Instrument(0, name="Piano accompaniment")
    strings = pretty_midi.Instrument(48, name="Strings chords")
    brass = pretty_midi.Instrument(56, name="Brass chords")
    drums = pretty_midi.Instrument(0, is_drum=True, name="Drums")
    scale = [72, 74, 76, 79, 76, 74, 72, 67]
    for i in range(int(seconds * 2)):
        start = i * 0.5
        melody.notes.append(pretty_midi.Note(96, scale[i % len(scale)] + offset, start, start + 0.5))
        drums.notes.append(pretty_midi.Note(60, 36 if i % 2 == 0 else 42, start, start + 0.1))
    for i in range(int(seconds)):
        root = (48 if i % 4 < 2 else 43) + offset
        bass.notes.append(pretty_midi.Note(75, root - 12, float(i), i + 1.0))
        for track, velocity in ((piano, 55), (strings, 45), (brass, 35)):
            for pitch in (root, root + 4, root + 7):
                track.notes.append(pretty_midi.Note(velocity, pitch, float(i), i + 1.0))
    midi.instruments = [melody, bass, piano, strings, brass, drums]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    midi.write(str(path))
    return midi


def create_demo(root):
    root = Path(root).resolve()
    for split_index, split in enumerate(("train", "validation", "test")):
        for i in range(2):
            make_midi(root / "source" / split / f"Demo{split_index}{i}" / "all_src.mid", split_index * 2 + i)
    soundfont = Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"
    if not soundfont.is_file():
        raise FileNotFoundError("The installed pretty_midi package does not contain TimGM6mb.sf2")
    configuration = {"input_root": "source", "output_root": "generated", "soundfont": str(soundfont),
                     "original_mode": "synthesize", "seed": 2026, "num_versions": 3,
                     "audio": {"sample_rate": 16000}}
    config_path = root / "demo.yaml"
    config_path.write_text(yaml.safe_dump(configuration, sort_keys=False), encoding="utf-8")
    return config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1] / "augmentation-demo"))
    parser.add_argument("--run", action="store_true")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    path = create_demo(args.root)
    print(f"Demo configuration: {path}")
    if args.run:
        from data.augmentation.config import load_config
        from data.augmentation.pipeline import generate_dataset
        summary = generate_dataset(load_config(path), workers=args.workers)
        print(f"{summary['status']}: {len(summary['tracks'])} source pieces; original + 3 augmented versions per piece")


if __name__ == "__main__":
    main()
