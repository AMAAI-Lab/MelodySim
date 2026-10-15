"""An explicit track-at-a-time pipeline, shared by serial and parallel runs."""

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import shutil
import uuid

import numpy as np
import soundfile as sf

from .audio import prepare_segments, read_audio, sample_audio_plan
from .manifest import derived_seed, output_file_record, run_provenance, stable_digest, verify_track, write_json
from .melody import identify_melody
from .midi import apply_midi_plan, sample_midi_plan
from .slakh import discover_tracks, file_sha256, read_midi
from .synth import available_presets, render_midi


@contextmanager
def output_lock(output):
    """An OS lock is released on process exit; a crash leaves no stale lock."""
    with (Path(output) / ".generation.lock").open("a+b") as stream:
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError(f"Another generation process is using {output}") from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _check_existing_splits(output, sources):
    previous_sources = []
    for path in output.glob("*/*/transformations.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
            previous_sources.append((manifest["track_key"].split("/", 1), manifest["source_hashes"]["midi_sha256"]))
        except (OSError, ValueError, KeyError):
            continue
    for source in sources:
        digest = file_sha256(source.midi_path)
        for (split, track_id), previous_digest in previous_sources:
            if split != source.split and (track_id == source.track_id or previous_digest == digest):
                raise ValueError(f"Source {source.key} overlaps an existing output source in split {split}")


def _safe_remove(path, parent):
    path, parent = Path(path).resolve(), Path(parent).resolve()
    if path == parent or not path.is_relative_to(parent):
        raise ValueError(f"Refusing to remove a directory outside {parent}")
    shutil.rmtree(path)


def process_track(source, config, provenance, override=None, overwrite=False, replay=None):
    source_hashes = {"midi_sha256": file_sha256(source.midi_path),
                     "audio_sha256": file_sha256(source.audio_path) if source.audio_path else None}
    fingerprint = stable_digest({"provenance": provenance, "source": source_hashes, "track": source.key, "melody_override": override})
    destination = Path(config.output_root) / source.split / source.track_id
    if not destination.resolve().is_relative_to(Path(config.output_root).resolve()):
        raise ValueError("Output track resolves outside output_root")
    if replay is not None:
        if replay["fingerprint"] != fingerprint or replay["track_key"] != source.key:
            raise ValueError("Replay requires matching source, configuration, model, soundfont and implementation")
    if destination.exists():
        marker = destination / "transformations.json"
        try:
            previous = json.loads(marker.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            previous = {}
        if previous.get("generated_by") != "MelodySim":
            raise ValueError(f"Refusing to replace an unrecognized output directory: {destination}")
        if previous.get("fingerprint") != fingerprint and not overwrite:
            raise ValueError(f"Output settings/source changed for {source.key}; choose a new output_root or --overwrite")
        if verify_track(destination, fingerprint) and not overwrite:
            return {"track_key": source.key, "status": "skipped", "fingerprint": fingerprint}
    midi = read_midi(source.midi_path)
    melody = identify_melody(midi, config.classifier_path, override)
    if replay and stable_digest(melody) != stable_digest(replay["melody"]):
        raise ValueError("Replay melody selection does not match identification on the original MIDI")
    programs, _ = available_presets(config.soundfont)
    sr = config.audio.sample_rate
    render_args = dict(soundfont=config.soundfont, sample_rate=sr, gain=config.synthesis.gain, tail_seconds=config.synthesis.tail_seconds)
    if source.audio_path:
        original, original_info = read_audio(source.audio_path, sr)
    else:
        original = render_midi(midi, **render_args)
        original_info = {"output_sample_rate": sr, "synthesized": True}
    if not np.any(original):
        raise ValueError(f"Original audio is silent: {source.key}")
    directory = destination.parent
    directory.mkdir(parents=True, exist_ok=True)
    staging = directory / f".{source.track_id}.staging-{uuid.uuid4().hex}"
    staging.mkdir()
    manifest = {"generated_by": "MelodySim", "schema_version": 1, "status": "complete",
                "track_key": source.key, "source": asdict(source), "source_hashes": source_hashes,
                "config": config.to_dict(), "provenance": provenance, "fingerprint": fingerprint,
                "melody": melody, "canonical_samples": len(original), "original_info": original_info,
                "versions": {}}
    try:
        for version_index in range(-1, config.num_versions):
            name = "original" if version_index < 0 else f"version_{version_index}"
            if version_index < 0:
                transformed, waveform, midi_plan, audio_plan = midi, original, None, None
            else:
                if replay:
                    midi_plan = replay["versions"][name]["midi_plan"]
                    audio_plan = replay["versions"][name]["audio_plan"]
                else:
                    midi_rng = np.random.default_rng(derived_seed(config.seed, source.key, name, "midi"))
                    audio_rng = np.random.default_rng(derived_seed(config.seed, source.key, name, "audio"))
                    midi_plan = sample_midi_plan(midi, melody["indices"], midi_rng, programs, config.midi)
                    audio_plan = sample_audio_plan(audio_rng, config.audio)
                transformed = apply_midi_plan(midi, midi_plan, melody_indices=melody["indices"], config=config.midi)
                waveform = render_midi(transformed, **render_args)
            midi_path = staging / f"{name}.mid"
            transformed.write(str(midi_path))
            version_dir = staging / name
            version_dir.mkdir()
            segments, files = [], []
            for segment, segment_info in prepare_segments(waveform, config.audio, audio_plan, len(original)):
                path = version_dir / f"{segment_info['index']}.wav"
                sf.write(path, segment, sr, subtype="PCM_16")
                segments.append(segment_info)
                files.append(output_file_record(path, staging, sr))
            manifest["versions"][name] = {"midi_plan": midi_plan, "audio_plan": audio_plan,
                                          "plan_sha256": stable_digest({"midi_plan": midi_plan, "audio_plan": audio_plan}),
                                          "rendered_samples": len(waveform), "canonical_adjustment_samples": len(original) - len(waveform),
                                          "segments": segments, "files": files,
                                          "midi_file": output_file_record(midi_path, staging)}
        write_json(staging / "transformations.json", manifest)
        if not verify_track(staging, fingerprint):
            raise ValueError(f"Generated output failed verification: {source.key}")
        backup = None
        if destination.exists():
            backup = directory / f".{source.track_id}.backup-{uuid.uuid4().hex}"
            destination.rename(backup)
        try:
            staging.rename(destination)
        except BaseException:
            if backup:
                backup.rename(destination)
            raise
        if backup:
            _safe_remove(backup, directory)
        return {"track_key": source.key, "status": "generated", "fingerprint": fingerprint}
    finally:
        if staging.exists():
            _safe_remove(staging, directory)


def generate_dataset(config, workers=1, limit=None, overwrite=False, dry_run=False, replay_path=None, progress=None):
    config.validate()
    if type(workers) is not int or workers < 1 or (limit is not None and (type(limit) is not int or limit < 1)):
        raise ValueError("workers and limit must be positive integers")
    for key in ("soundfont", "classifier_path"):
        if not Path(getattr(config, key)).is_file():
            raise FileNotFoundError(f"Missing {key}: {getattr(config, key)}")
    sources = discover_tracks(config)
    known_sources = {source.key for source in sources}
    total_sources = len(sources)
    if limit is not None:
        sources = sources[:limit]
    overrides = {}
    if config.melody_overrides:
        overrides = json.loads(Path(config.melody_overrides).read_text(encoding="utf-8"))
        if not isinstance(overrides, dict):
            raise ValueError("melody_overrides must be a JSON object mapping split/track IDs to index lists")
        if set(overrides) - known_sources:
            raise ValueError(f"Unknown melody override source IDs: {sorted(set(overrides) - known_sources)}")
        for key, indices in overrides.items():
            if not isinstance(indices, list) or not indices or any(type(i) is not int or i < 0 for i in indices) or len(set(indices)) != len(indices):
                raise ValueError(f"Melody override for {key} must be a nonempty list of unique nonnegative integer indices")
    if dry_run:
        return {"status": "dry_run", "discovered_sources": total_sources, "selected_sources": len(sources), "tracks": [source.key for source in sources]}
    replay = None
    if replay_path:
        replay = json.loads(Path(replay_path).read_text(encoding="utf-8"))
        if not isinstance(replay, dict) or replay.get("generated_by") != "MelodySim" or replay.get("schema_version") != 1 or not {"fingerprint", "versions", "melody"} <= replay.keys():
            raise ValueError("Replay file is not a complete MelodySim augmentation manifest")
        expected_versions = {"original", *(f"version_{i}" for i in range(config.num_versions))}
        if replay.get("status") != "complete" or not isinstance(replay["versions"], dict) or set(replay["versions"]) != expected_versions:
            raise ValueError("Replay manifest is incomplete or has the wrong number of versions")
        for settings in replay["versions"].values():
            if not isinstance(settings, dict) or not {"midi_plan", "audio_plan", "plan_sha256"} <= settings.keys() or settings["plan_sha256"] != stable_digest({"midi_plan": settings["midi_plan"], "audio_plan": settings["audio_plan"]}):
                raise ValueError("Replay augmentation plan checksum does not match")
        sources = [source for source in sources if source.key == replay.get("track_key")]
        if len(sources) != 1:
            raise ValueError("Replay manifest must identify exactly one selected input source")
    provenance = run_provenance(config)
    output = Path(config.output_root)
    output.mkdir(parents=True, exist_ok=True)
    with output_lock(output):
        _check_existing_splits(output, sources)
        return _run_jobs(output, sources, config, provenance, overrides, overwrite, replay, workers, total_sources, progress)


def _run_jobs(output, sources, config, provenance, overrides, overwrite, replay, workers, total_sources, progress):
    summary = {"generated_by": "MelodySim", "schema_version": 1, "status": "running",
               "provenance": provenance, "discovered_sources": total_sources, "selected_sources": len(sources),
               "selection": [source.key for source in sources], "tracks": [], "errors": []}
    write_json(output / "dataset_manifest.json", summary)
    jobs = [(source, config, provenance, overrides.get(source.key), overwrite, replay) for source in sources]
    if workers == 1:
        for job in jobs:
            try:
                record = process_track(*job)
                summary["tracks"].append(record)
                if progress:
                    progress(record)
            except Exception as error:
                summary["errors"].append({"track_key": job[0].key, "error": f"{type(error).__name__}: {error}"})
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            pending = {executor.submit(process_track, *job): job[0].key for job in jobs}
            for future in as_completed(pending):
                try:
                    record = future.result()
                    summary["tracks"].append(record)
                    if progress:
                        progress(record)
                except Exception as error:
                    summary["errors"].append({"track_key": pending[future], "error": f"{type(error).__name__}: {error}"})
    summary["tracks"].sort(key=lambda record: record["track_key"])
    summary["errors"].sort(key=lambda record: record["track_key"])
    summary["status"] = "failed" if summary["errors"] else "complete"
    write_json(output / "dataset_manifest.json", summary)
    if summary["errors"]:
        details = "; ".join(f"{error['track_key']}: {error['error']}" for error in summary["errors"])
        raise RuntimeError(f"Dataset generation failed: {details}. See {output / 'dataset_manifest.json'}")
    return summary
