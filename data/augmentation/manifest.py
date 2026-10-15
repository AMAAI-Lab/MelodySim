"""Generation provenance, deterministic identities and verified resume."""

import hashlib
from importlib.metadata import PackageNotFoundError, version
import json
from pathlib import Path
import platform

import soundfile as sf
import numpy as np

from . import POLICY_VERSION
from .slakh import file_sha256
from .synth import backend_identity


def _json_scalar(value):
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def stable_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False, default=_json_scalar).encode()).hexdigest()


def derived_seed(seed, track_key, version_name, stage):
    return int(stable_digest([seed, track_key, version_name, stage])[:16], 16)


def write_json(path, content):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(content, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False, default=_json_scalar)
        stream.write("\n")
    temporary.replace(path)


def run_provenance(config):
    packages = {}
    for name in ("numpy", "scipy", "pretty-midi", "scikit-learn", "sf2-loader", "musicpy", "librosa", "soundfile"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "missing"
    sources = {path.name: file_sha256(path) for path in sorted(Path(__file__).parent.glob("*.py"))}
    effective = config.to_dict()
    for key in ("input_root", "output_root", "soundfont", "classifier_path", "melody_overrides"):
        effective.pop(key, None)
    return {"policy_version": POLICY_VERSION, "effective_config": effective, "packages": packages,
            "python_version": platform.python_version(), "platform": platform.system(),
            "implementation_sha256": stable_digest(sources), "soundfont_sha256": file_sha256(config.soundfont),
            "classifier_sha256": file_sha256(config.classifier_path), "synthesis_backend": backend_identity()}


def output_file_record(path, directory, sample_rate=None):
    record = {"path": Path(path).relative_to(directory).as_posix(), "sha256": file_sha256(path), "bytes": Path(path).stat().st_size}
    if sample_rate is not None:
        info = sf.info(path)
        if info.samplerate != sample_rate or info.channels != 1 or not info.frames:
            raise ValueError(f"Invalid generated audio: {path}")
        record.update(sample_rate=info.samplerate, samples=info.frames, channels=info.channels)
    return record


def verify_track(directory, fingerprint=None):
    directory = Path(directory)
    try:
        manifest = json.loads((directory / "transformations.json").read_text(encoding="utf-8"))
        if manifest.get("generated_by") != "MelodySim" or manifest.get("schema_version") != 1 or manifest.get("status") != "complete":
            return False
        if fingerprint and manifest.get("fingerprint") != fingerprint:
            return False
        expected = {"original", *(f"version_{i}" for i in range(manifest["config"]["num_versions"]))}
        if set(manifest["versions"]) != expected:
            return False
        counts = set()
        for settings in manifest["versions"].values():
            if settings.get("plan_sha256") != stable_digest({"midi_plan": settings["midi_plan"], "audio_plan": settings["audio_plan"]}):
                return False
            counts.add(len(settings["segments"]))
            for record in [*settings["files"], settings["midi_file"]]:
                path = (directory / record["path"]).resolve()
                if not path.is_relative_to(directory.resolve()) or not path.is_file() or path.stat().st_size != record["bytes"] or file_sha256(path) != record["sha256"]:
                    return False
                if "sample_rate" in record:
                    info = sf.info(path)
                    if (info.frames, info.samplerate, info.channels) != (record["samples"], record["sample_rate"], 1):
                        return False
            if len(settings["files"]) != len(settings["segments"]):
                return False
        return len(counts) == 1 and min(counts) > 0
    except (OSError, ValueError, KeyError, TypeError, RuntimeError):
        return False
