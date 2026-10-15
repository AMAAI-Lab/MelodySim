"""Discover sources without mixing source-piece splits."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import re

import pretty_midi


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class TrackSource:
    track_id: str
    split: str
    midi_path: str
    audio_path: str | None

    @property
    def key(self):
        return f"{self.split}/{self.track_id}"


def discover_tracks(config):
    root = Path(config.input_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {root}")
    if config.split:
        groups = [(config.split, root)]
    else:
        groups = [(name, root / name) for name in config.splits if (root / name).is_dir()]
        if not groups:
            raise ValueError("No split directories found; set split explicitly for a flat directory")
    sources = []
    identities, hashes = {}, {}
    for split, directory in groups:
        for path in sorted(directory.iterdir()):
            if path.name.startswith("."):
                continue
            if path.is_dir():
                midi = path / config.midi_filename
                if not midi.is_file():
                    raise FileNotFoundError(f"Missing MIDI source: {midi}")
                track_id = path.name
                audio = path / config.audio_filename
            elif path.suffix.lower() in (".mid", ".midi"):
                midi, track_id = path, path.stem
                audio = path.with_suffix(".wav")
            else:
                continue
            if not re.fullmatch(r"[A-Za-z0-9_-]+", track_id):
                raise ValueError(f"Track ID {track_id!r} must use letters, digits, underscores or hyphens")
            digest = file_sha256(midi)
            for mapping, identity in ((identities, track_id), (hashes, digest)):
                previous = mapping.get(identity)
                if previous is not None and previous != split:
                    raise ValueError(f"Source {track_id} appears across splits {previous} and {split}")
                mapping[identity] = split
            if config.original_mode == "audio_mix" and not audio.is_file():
                raise FileNotFoundError(f"Missing original audio: {audio}; use original_mode: synthesize for MIDI-only input")
            sources.append(TrackSource(track_id, split, str(midi.resolve()), str(audio.resolve()) if config.original_mode == "audio_mix" else None))
    if not sources:
        raise ValueError("No MIDI sources found")
    keys = [source.key for source in sources]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate track IDs within a split")
    return sources


def read_midi(path):
    midi = pretty_midi.PrettyMIDI(str(path))
    for track in midi.instruments:
        track.notes.sort(key=lambda n: (n.start, n.pitch, n.end))
        for note in track.notes:
            if not 0 <= note.start < note.end or not 0 <= note.pitch <= 127:
                raise ValueError(f"Invalid note in {path}")
    if not any(track.notes and not track.is_drum for track in midi.instruments):
        raise ValueError(f"MIDI has no pitched notes: {path}")
    return midi
