"""Run with python -m data.generate_dataset --config data/augmentation/configs/slakh.yaml."""

import argparse
import json
import sys

from .augmentation.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description="Generate reproducible melody-aware MIDI/audio variants")
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root")
    parser.add_argument("--output-root")
    parser.add_argument("--soundfont")
    parser.add_argument("--split", choices=("train", "validation", "test"))
    parser.add_argument("--original-mode", choices=("audio_mix", "synthesize"))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--num-versions", type=int)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--replay", help="Replay one track's transformations.json with matching settings/assets")
    args = parser.parse_args(argv)
    try:
        from .augmentation.pipeline import generate_dataset
        config = load_config(args.config, **{name: getattr(args, name) for name in ("input_root", "output_root", "soundfont", "split", "original_mode", "seed", "num_versions")})
        summary = generate_dataset(config, workers=args.workers, limit=args.limit, overwrite=args.overwrite,
                                   dry_run=args.dry_run, replay_path=args.replay,
                                   progress=lambda record: print(f"{record['status']}: {record['track_key']}", flush=True))
    except (ValueError, FileNotFoundError, RuntimeError, OSError) as error:
        print(str(error), file=sys.stderr)
        return 1
    print(json.dumps({key: summary[key] for key in ("status", "discovered_sources", "selected_sources")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
