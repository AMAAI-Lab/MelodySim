"""Check an augmentation installation, including real synthesis and audio processing.

Run from the repository root: python -m tools.check_augmentation
"""

import argparse
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform
import struct
import sys


ROOT = Path(__file__).resolve().parents[1]


def check_dependencies():
    if sys.version_info[:2] != (3, 12) or struct.calcsize("P") != 8:
        raise RuntimeError("Use 64-bit CPython 3.12 in a separate augmentation environment")
    # Keep the diagnostic runnable even when no project dependency is installed.
    from packaging.requirements import Requirement

    installed = {}
    failures = []
    for line in (ROOT / "requirements-augmentation.txt").read_text(encoding="utf-8").splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        requirement = Requirement(line)
        try:
            installed[requirement.name] = version(requirement.name)
        except PackageNotFoundError:
            failures.append(f"{requirement.name}: missing")
            continue
        if not requirement.specifier.contains(installed[requirement.name]):
            failures.append(f"{requirement.name}=={installed[requirement.name]} does not satisfy {requirement.specifier}")
    if failures:
        raise RuntimeError("; ".join(failures))
    return installed


def validate_soundfont(path):
    path = Path(path).expanduser().resolve()
    with path.open("rb") as stream:
        header = stream.read(12)
    if path.suffix.lower() != ".sf2" or len(header) != 12 or header[:4] != b"RIFF" or header[8:] != b"sfbk":
        raise ValueError("Expected an extracted RIFF SoundFont .sf2 file, not .sfpack/.sfArk/.zip, HTML or MIDI.js samples")
    if struct.unpack("<I", header[4:8])[0] + 8 > path.stat().st_size:
        raise ValueError("SoundFont is truncated; download and extract it again")
    return path


def check_runtime(soundfont, classifier_path, output_dir):
    import numpy as np
    import pretty_midi
    import soundfile as sf

    from data.augmentation.audio import prepare_segments, read_audio
    from data.augmentation.config import AudioConfig
    from data.augmentation.melody import identify_melody
    from data.augmentation.slakh import file_sha256
    from data.augmentation.synth import available_presets, backend_identity, render_midi
    from tools.create_augmentation_demo import make_midi

    soundfont = validate_soundfont(soundfont)
    melodic, drums = available_presets(str(soundfont))
    missing = sorted(set(range(128)) - set(melodic))
    if missing or 0 not in drums:
        raise ValueError(f"Use a complete GM SoundFont: missing bank-0 programs {missing}; bank-128 drum kit 0 present: {0 in drums}")
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    midi_path = output_dir / "self_check.mid"
    make_midi(midi_path, seconds=4.0)
    midi = pretty_midi.PrettyMIDI(str(midi_path))
    melody = identify_melody(midi, str(classifier_path))
    waveform = render_midi(midi, str(soundfont), sample_rate=16000)
    config = AudioConfig(sample_rate=16000, window_seconds=2.0, hop_seconds=2.0)
    plan = {"pitch_shift_semitones": 2, "time_shift_samples": 1600,
            "tempo_factor": 1.05, "tempo_mode": "stretch"}
    processed, _ = next(prepare_segments(waveform, config, plan))
    wav_path = output_dir / "self_check.wav"
    sf.write(wav_path, processed, config.sample_rate, subtype="PCM_16")
    restored, _ = read_audio(wav_path, config.sample_rate)
    # Exercise the FLAC reader required by Slakh, without requiring ffmpeg.
    flac_path = output_dir / "self_check.flac"
    sf.write(flac_path, processed, config.sample_rate, subtype="PCM_16")
    flac, _ = read_audio(flac_path, config.sample_rate)
    # libsndfile's WAV and FLAC float-to-PCM16 conversions can differ by one LSB.
    if not np.allclose(restored, flac, atol=1 / 32768, rtol=0) or np.max(np.abs(restored)) <= 1e-4:
        raise RuntimeError("WAV/FLAC round-trip produced silent or inconsistent audio")
    return {"soundfont": str(soundfont), "soundfont_sha256": file_sha256(soundfont),
            "classifier_sha256": file_sha256(classifier_path), "melody_indices": melody["indices"],
            "melodic_presets": len(melodic), "drum_presets": len(drums),
            "synthesis_backend": backend_identity(), "sample_rate": config.sample_rate,
            "output_samples": len(restored), "wav": str(wav_path)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--soundfont", help="Extracted GM .sf2 file; relative to the current working directory")
    selection.add_argument("--config", help="Use the SoundFont/classifier from a generation YAML")
    parser.add_argument("--output-dir", default="outputs/augmentation-check",
                        help="Diagnostic files are replaced on rerun; default: %(default)s")
    args = parser.parse_args(argv)
    print(f"Python: {sys.executable} ({platform.python_version()}, {platform.machine()})", flush=True)
    try:
        packages = check_dependencies()
        print("[OK] Python and required package versions", flush=True)
        from data.augmentation.config import ASSET_DIR, load_config
        import pretty_midi

        classifier = ASSET_DIR / "classifier_model.pkl"
        if args.config:
            config = load_config(args.config)
            soundfont, classifier = config.soundfont, config.classifier_path
        else:
            soundfont = args.soundfont or Path(pretty_midi.__file__).parent / "TimGM6mb.sf2"
        result = check_runtime(soundfont, classifier, args.output_dir)
        result.update(status="complete", python=platform.python_version(), platform=platform.platform(), packages=packages)
        report = Path(args.output_dir).resolve() / "report.json"
        report.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        print(f"[OK] Classifier, GM presets and FluidSynth {result['synthesis_backend']['fluidsynth_version']}", flush=True)
        print("[OK] Pitch shift, time shift, tempo stretch and WAV/FLAC round-trip", flush=True)
        print(f"[OK] SoundFont SHA-256: {result['soundfont_sha256']}", flush=True)
        print(f"[OK] Report: {report}", flush=True)
        return 0
    except Exception as error:
        print(f"[FAIL] {type(error).__name__}: {error}", file=sys.stderr)
        print("See docs/augmentation-setup.md#troubleshooting; install dependencies into the Python shown above.", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
