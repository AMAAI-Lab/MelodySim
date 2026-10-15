"""Validate source-piece identity before constructing feature datasets."""

from pathlib import Path


def feature_split_directories(root):
    root = Path(root)
    directories = {name: root / name for name in ("train", "validation", "test")}
    owners = {}
    for name, directory in directories.items():
        if not directory.is_dir():
            if name == "test":
                continue
            raise ValueError(f"Missing feature split directory: {directory}")
        tracks = sorted(directory.glob("*.npy"))
        if name in ("train", "validation") and len(tracks) < 2:
            raise ValueError(f"{directory} needs at least two source pieces for negative sampling")
        for track in tracks:
            if track.stem in owners:
                raise ValueError(f"Source piece {track.stem} occurs in both {owners[track.stem]} and {name}")
            owners[track.stem] = name
    return directories
