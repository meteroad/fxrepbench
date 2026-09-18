from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__
from .manifests import get_item, load_tracks, validate_track


def _track_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--track",
        choices=tuple(sorted(load_tracks())),
        default="counterfx-200",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fxrepbench",
        description="Reconstruct, evaluate, and reproduce FXRepBench tracks.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("info", help="List benchmark tracks.")

    validate = commands.add_parser("validate", help="Validate frozen manifests.")
    validate.add_argument("--track", choices=tuple(sorted(load_tracks())))

    render = commands.add_parser("render-item", help="Reconstruct one CounterFX item.")
    _track_arg(render)
    render.add_argument("--item-id", required=True)
    render.add_argument("--data-root", type=Path, required=True)
    render.add_argument("--output-dir", type=Path, required=True)
    render.add_argument("--skip-source-hash", action="store_true")

    prepare = commands.add_parser("prepare", help="Reconstruct every item in a track.")
    _track_arg(prepare)
    prepare.add_argument("--data-root", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--skip-source-hash", action="store_true")

    evaluate = commands.add_parser(
        "evaluate",
        help="Evaluate system outputs or a ranked candidate pool.",
    )
    _track_arg(evaluate)
    evaluate.add_argument("--submission", type=Path, required=True)
    evaluate.add_argument("--data-root", type=Path, required=True)
    evaluate.add_argument("--output-dir", type=Path, required=True)
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--batch-size", type=int, default=8)
    evaluate.add_argument("--allow-partial", action="store_true")
    evaluate.add_argument("--skip-source-hash", action="store_true")

    pool = commands.add_parser(
        "generate-random-pool",
        help="Generate the deterministic shared Global Random candidate pool.",
    )
    _track_arg(pool)
    pool.add_argument("--topology", choices=("known", "hidden"), required=True)
    pool.add_argument("--budget", type=int, default=512)
    pool.add_argument("--item-id")
    pool.add_argument("--data-root", type=Path, required=True)
    pool.add_argument("--output", type=Path, required=True)
    pool.add_argument("--audio-dir", type=Path)
    pool.add_argument("--seed", type=int)
    pool.add_argument("--skip-source-hash", action="store_true")
    return parser


def _info() -> int:
    print(f"FXRepBench {__version__}")
    for track in load_tracks().values():
        print(f"{track.key:14s} {track.role:11s} {track.expected_items:3d} items")
    return 0


def _validate(track_key: str | None) -> int:
    tracks = load_tracks()
    selected = [tracks[track_key]] if track_key else list(tracks.values())
    failed = False
    for track in selected:
        report = validate_track(track)
        if report["errors"]:
            failed = True
            print(json.dumps(report, indent=2, sort_keys=True))
        else:
            print(
                f"PASS {track.key}: {report['audio_items']} audio rows, "
                f"{report['target_items']} target rows"
            )
    return 1 if failed else 0


def _render(args: argparse.Namespace) -> int:
    try:
        from .audio import reconstruct_item, write_item
    except ImportError as exc:
        raise SystemExit(
            "Install reference support with: python -m pip install -e '.[reference]'"
        ) from exc

    track = load_tracks()[args.track]
    audio_row, target_row = get_item(track, args.item_id)
    source, reference, target, sample_rate = reconstruct_item(
        audio_row,
        target_row,
        args.data_root,
        verify_hash=not args.skip_source_hash,
    )
    write_item(args.output_dir, source, reference, target, sample_rate)
    print(f"Wrote {args.item_id} to {args.output_dir}")
    return 0


def _prepare(args: argparse.Namespace) -> int:
    try:
        from .audio import prepare_track
    except ImportError as exc:
        raise SystemExit(
            "Install reference support with: python -m pip install -e '.[reference]'"
        ) from exc

    summary = prepare_track(
        load_tracks()[args.track],
        args.data_root,
        args.output_dir,
        verify_hash=not args.skip_source_hash,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _evaluate(args: argparse.Namespace) -> int:
    try:
        from .evaluation import evaluate_submission
    except ImportError as exc:
        raise SystemExit(
            "Install reference support with: python -m pip install -e '.[reference]'"
        ) from exc

    summary = evaluate_submission(
        load_tracks()[args.track],
        args.submission,
        args.data_root,
        args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        allow_partial=args.allow_partial,
        verify_source_hash=not args.skip_source_hash,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _generate_random_pool(args: argparse.Namespace) -> int:
    try:
        from .candidates import generate_random_pool
    except ImportError as exc:
        raise SystemExit(
            "Install reference support with: python -m pip install -e '.[reference]'"
        ) from exc

    summary = generate_random_pool(
        load_tracks()[args.track],
        args.data_root,
        args.output,
        topology=args.topology,
        budget=args.budget,
        item_id=args.item_id,
        audio_dir=args.audio_dir,
        seed=args.seed,
        verify_source_hash=not args.skip_source_hash,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "info":
        raise SystemExit(_info())
    if args.command == "validate":
        raise SystemExit(_validate(args.track))
    if args.command == "render-item":
        raise SystemExit(_render(args))
    if args.command == "prepare":
        raise SystemExit(_prepare(args))
    if args.command == "evaluate":
        raise SystemExit(_evaluate(args))
    if args.command == "generate-random-pool":
        raise SystemExit(_generate_random_pool(args))
    raise AssertionError(args.command)
