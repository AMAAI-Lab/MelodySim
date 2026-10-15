from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import yaml

from data.augmentation.config import load_config
from data.augmentation.manifest import verify_track
from data.augmentation.pipeline import generate_dataset, output_lock
from data.augmentation.slakh import discover_tracks
from data.augmentation.synth import render_midi
from tools.create_augmentation_demo import make_midi


def _audio_hashes(root):
    manifests = {}
    for path in Path(root).glob("*/*/transformations.json"):
        manifest = json.loads(path.read_text())
        manifests[manifest["track_key"]] = {name: [r["sha256"] for r in settings["files"]] for name, settings in manifest["versions"].items()}
    return manifests


def test_discovery_prevents_source_leakage(config):
    duplicate = Path(config.input_root) / "test" / "Other" / "all_src.mid"
    duplicate.parent.mkdir(parents=True)
    duplicate.write_bytes((Path(config.input_root) / "train/TrackA/all_src.mid").read_bytes())
    with pytest.raises(ValueError, match="across splits"):
        discover_tracks(config)


def test_configuration_validation_and_relative_paths(config, tmp_path):
    path = tmp_path / "config.yaml"
    content = config.to_dict()
    content.update(input_root="source", output_root="generated")
    path.write_text(yaml.safe_dump(content))
    loaded = load_config(path)
    assert loaded.input_root == str(tmp_path / "source")
    content["audio"]["tempo_range"] = [0, 1]
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="tempo_range"):
        load_config(path)
    content["audio"]["tempo_range"] = [0.9, 1.1]
    content["typo"] = 1
    path.write_text(yaml.safe_dump(content))
    with pytest.raises(ValueError, match="Unknown"):
        load_config(path)


def test_dry_run_does_not_create_outputs(config):
    assert generate_dataset(config, dry_run=True)["selected_sources"] == 2
    assert not Path(config.output_root).exists()


@pytest.mark.parametrize("indices", [None, [], [True], [0, 0], [-1]])
def test_invalid_manual_melody_override_never_falls_back_silently(config, tmp_path, indices):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"train/TrackA": indices}))
    config.melody_overrides = str(path)
    with pytest.raises(ValueError, match="Melody override"):
        generate_dataset(config, dry_run=True)


def test_unknown_melody_override_source_is_rejected(config, tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"train/Typo": [0]}))
    config.melody_overrides = str(path)
    with pytest.raises(ValueError, match="Unknown melody override"):
        generate_dataset(config, dry_run=True)


def test_limited_run_accepts_overrides_for_unselected_valid_sources(config, tmp_path):
    path = tmp_path / "overrides.json"
    path.write_text(json.dumps({"train/TrackB": [0]}))
    config.melody_overrides = str(path)
    assert generate_dataset(config, limit=1, dry_run=True)["selected_sources"] == 1


def test_output_lock_prevents_concurrent_runs(config):
    output = Path(config.output_root)
    output.mkdir()
    with output_lock(output):
        with pytest.raises(RuntimeError, match="Another generation"):
            generate_dataset(config)
    with output_lock(output):
        pass  # The OS releases the lock; no manual stale-lock recovery is needed.


def test_existing_output_cannot_leak_across_splits(config):
    source = discover_tracks(config)[0]
    from data.augmentation.slakh import file_sha256
    previous = Path(config.output_root) / "test/Other"
    previous.mkdir(parents=True)
    (previous / "transformations.json").write_text(json.dumps({"track_key": "test/Other", "source_hashes": {"midi_sha256": file_sha256(source.midi_path)}}))
    with pytest.raises(ValueError, match="existing output"):
        generate_dataset(config)


@pytest.mark.integration
def test_real_soundfont_sample_rate_and_controls(config, midi):
    midi.instruments = [midi.instruments[0]]
    midi.instruments[0].control_changes = []
    for rate in (8000, 16000):
        audio = render_midi(midi, config.soundfont, rate)
        assert len(audio) == round((midi.get_end_time() + 0.5) * rate)
        assert np.max(np.abs(audio)) > 0.001
    midi.instruments[0].control_changes = [__import__("pretty_midi").ControlChange(7, 0, 0)]
    with pytest.raises(ValueError, match="silent"):
        render_midi(midi, config.soundfont, 8000)


@pytest.mark.integration
def test_end_to_end_resume_repair_replay_and_parallel(config, tmp_path):
    summary = generate_dataset(config)
    assert summary["status"] == "complete"
    assert all(track["status"] == "generated" for track in summary["tracks"])
    first = _audio_hashes(config.output_root)
    for directory in Path(config.output_root).glob("*/*"):
        assert verify_track(directory)
        manifest = json.loads((directory / "transformations.json").read_text())
        assert len(manifest["versions"]) == 3
        counts = {len(version["segments"]) for version in manifest["versions"].values()}
        assert counts == {2}
        for path in directory.glob("*/*.wav"):
            info = sf.info(path)
            assert info.channels == 1 and info.samplerate == config.audio.sample_rate
    assert all(track["status"] == "skipped" for track in generate_dataset(config)["tracks"])
    damaged = next(Path(config.output_root).glob("*/*/version_0/0.wav"))
    damaged.write_bytes(b"broken")
    assert not verify_track(damaged.parent.parent)
    assert any(track["status"] == "generated" for track in generate_dataset(config)["tracks"])
    assert first == _audio_hashes(config.output_root)
    manifest_path = Path(config.output_root) / "train/TrackA/transformations.json"
    replay_config = deepcopy(config)
    replay_config.output_root = str(tmp_path / "replayed")
    assert generate_dataset(replay_config, replay_path=manifest_path)["selected_sources"] == 1
    assert _audio_hashes(replay_config.output_root)["train/TrackA"] == first["train/TrackA"]
    parallel_config = deepcopy(config)
    parallel_config.output_root = str(tmp_path / "parallel")
    generate_dataset(parallel_config, workers=2)
    assert first == _audio_hashes(parallel_config.output_root)
    changed = deepcopy(config)
    changed.seed += 1
    with pytest.raises(RuntimeError, match="changed"):
        generate_dataset(changed)
    assert first == _audio_hashes(config.output_root)


@pytest.mark.integration
def test_original_audio_is_resampled_and_short_piece_is_padded(config):
    source = Path(config.input_root) / "train/TrackA/mix.flac"
    tone = np.sin(2 * np.pi * 220 * np.arange(4410) / 4410).astype(np.float32) * 0.1
    sf.write(source, np.stack([tone, tone], axis=1), 4410)
    config.original_mode = "audio_mix"
    # Restrict discovery to a single split directory containing one track.
    config.input_root = str(source.parent.parent)
    config.split = "train"
    other = Path(config.input_root) / "TrackB/mix.flac"
    sf.write(other, tone, 4410)
    generate_dataset(config, limit=1)
    manifest = json.loads((Path(config.output_root) / "train/TrackA/transformations.json").read_text())
    assert manifest["original_info"]["source_sample_rate"] == 4410
    assert manifest["canonical_samples"] == 8000
    for version in manifest["versions"].values():
        assert len(version["segments"]) == 1


@pytest.mark.integration
def test_plan_integrity_and_melody_guards_on_replay(config, tmp_path):
    from data.augmentation.manifest import stable_digest
    config.num_versions = 1
    generate_dataset(config, limit=1)
    track_dir = Path(config.output_root) / "train/TrackA"
    manifest = json.loads((track_dir / "transformations.json").read_text())
    selected = manifest["melody"]["indices"][0]
    settings = manifest["versions"]["version_0"]
    settings["midi_plan"]["tracks"][selected]["removed"] = True
    (track_dir / "transformations.json").write_text(json.dumps(manifest))
    assert not verify_track(track_dir)
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(manifest))
    replay_config = deepcopy(config)
    replay_config.output_root = str(tmp_path / "replayed")
    with pytest.raises(ValueError, match="checksum"):
        generate_dataset(replay_config, replay_path=replay_path)
    # Even an explicitly edited, re-checksummed plan must preserve true roles.
    settings["plan_sha256"] = stable_digest({"midi_plan": settings["midi_plan"], "audio_plan": settings["audio_plan"]})
    replay_path.write_text(json.dumps(manifest))
    with pytest.raises(RuntimeError, match="protected"):
        generate_dataset(replay_config, replay_path=replay_path)
    changed_melody = deepcopy(manifest)
    changed_melody["melody"]["indices"] = [1 if selected != 1 else 0]
    replay_path.write_text(json.dumps(changed_melody))
    with pytest.raises(RuntimeError, match="melody selection"):
        generate_dataset(replay_config, replay_path=replay_path)
