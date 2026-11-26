"""Command-line entry points. Each command is a thin wrapper around package functions.

Run ``python -m cuneiform.cli --help`` (or ``cuneiform --help``) for the list.
Typical order: fetch-metadata -> build-manifest -> prepare -> train -> evaluate.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from cuneiform.config import DEFAULT_CONFIG_PATH, ProjectConfig, load_config

log = logging.getLogger("cuneiform")


def _cmd_fetch_metadata(config: ProjectConfig, args: argparse.Namespace) -> None:
    from cuneiform.data import heicubeda
    from cuneiform.data.prepare import label_summary

    heicubeda.download_metadata(config.paths.metadata_json)
    print(f"metadata: {config.paths.metadata_json}")
    tablets = heicubeda.build_zip_index(config.paths.zip_index)
    print(f"zip index: {config.paths.zip_index} ({len(tablets)} scans)")
    print("\nAll HeiCuBeDa tablets by CDLI period:")
    for period, count in label_summary(config).items():
        mapped = config.label_for_period(period)
        print(f"  {count:5d}  {period}" + (f"   -> {mapped}" if mapped else ""))


def _cmd_build_manifest(config: ProjectConfig, args: argparse.Namespace) -> None:
    from cuneiform.data.prepare import build_heicubeda_manifest

    rows = build_heicubeda_manifest(config)
    print(f"wrote {len(rows)} rows to {config.paths.manifest}")
    _cmd_summarize(config, args)


def _cmd_assign_splits(config: ProjectConfig, args: argparse.Namespace) -> None:
    from cuneiform.data.manifest import read_manifest, validate_manifest, write_manifest
    from cuneiform.data.splits import assign_splits, check_split_isolation

    rows = read_manifest(config.paths.manifest)
    validate_manifest(rows, config.class_names)
    rows = assign_splits(rows, config.dataset.split_fractions, config.dataset.seed)
    check_split_isolation(rows)
    write_manifest(rows, config.paths.manifest)
    _cmd_summarize(config, args)


def _cmd_summarize(config: ProjectConfig, args: argparse.Namespace) -> None:
    from cuneiform.data.cache import DepthMapCache
    from cuneiform.data.manifest import class_counts, read_manifest

    rows = read_manifest(config.paths.manifest)
    cache = DepthMapCache(config.paths.processed_dir, config.preprocessing)
    invalid = cache.invalid_ids()
    cached_rows = [r for r in rows if cache.has(r.tablet_id)]
    summary = {
        "manifest_rows": class_counts(rows),
        "with_depth_map": class_counts(cached_rows),
        "invalid_scans": len(invalid & {r.tablet_id for r in rows}),
        "cache_dir": str(cache.directory),
    }
    report = config.paths.artifacts_dir / "reports" / "data_summary.json"
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(summary, indent=2))
    print(f"\n{'period':<18}{'split':<8}{'manifest':>10}{'depth map':>11}")
    for period in config.class_names:
        for split in ("train", "val", "test"):
            total = summary["manifest_rows"].get(period, {}).get(split, 0)
            done = summary["with_depth_map"].get(period, {}).get(split, 0)
            print(f"{period:<18}{split:<8}{total:>10}{done:>11}")
    print(f"invalid scans logged: {summary['invalid_scans']}  (see {cache.invalid_log})")
    print(f"summary written to {report}")


def _cmd_prepare(config: ProjectConfig, args: argparse.Namespace) -> None:
    from cuneiform.data.prepare import prepare_depth_maps

    report = prepare_depth_maps(config, workers=args.workers, limit=args.limit,
                                retry_invalid=args.retry_invalid, tablet_ids=args.tablet,
                                keep_raw=args.keep_raw)
    print(report.summary())
    for tablet_id, reason in report.invalid.items():
        print(f"  invalid {tablet_id}: {reason}")
    if report.failed:
        print("Some tablets failed with transient errors; re-run the same command to retry them.")


def _cmd_preview(config: ProjectConfig, args: argparse.Namespace) -> None:
    from cuneiform.data.cache import DepthMapCache
    from cuneiform.geometry.depth_map import scan_to_face_depth_maps
    from cuneiform.visualization import save_depth_preview

    if args.ply:
        maps = scan_to_face_depth_maps(Path(args.ply), config.preprocessing.resolution,
                                       config.preprocessing.margin_fraction)
        name = Path(args.ply).stem
    else:
        maps = DepthMapCache(config.paths.processed_dir, config.preprocessing).load(args.tablet_id)
        name = args.tablet_id
    out = config.paths.artifacts_dir / "previews" / f"{name}.png"
    save_depth_preview(maps, out, title=name)
    print(f"saved {out}")


def _cmd_train(config: ProjectConfig, args: argparse.Namespace) -> None:
    from cuneiform.modeling.train import train_model

    directory = train_model(config, run_name=args.run_name, epochs=args.epochs)
    print(f"model bundle: {directory}")
    if not args.no_evaluate:
        from cuneiform.modeling.evaluate import evaluate_bundle

        evaluate_bundle(directory, config, split="val")


def _cmd_evaluate(config: ProjectConfig, args: argparse.Namespace) -> None:
    from cuneiform.modeling.bundle import models_root, resolve_bundle_dir
    from cuneiform.modeling.evaluate import evaluate_bundle

    directory = resolve_bundle_dir(models_root(config), args.run)
    metrics = evaluate_bundle(directory, config, split=args.split)
    model, base = metrics["model"], metrics["majority_baseline"]
    print(f"\n{args.split} split: {metrics['num_tablets']} tablets, support {metrics['support']}")
    print(f"  model             accuracy {model['accuracy']:.3f}   macro F1 {model['macro_f1']:.3f}")
    print(f"  majority baseline accuracy {base['accuracy']:.3f}   macro F1 {base['macro_f1']:.3f}  (always '{base['predicts']}')")
    print(f"  confusion matrix (rows = true {metrics['class_names']}): {model['confusion_matrix']}")
    print(f"  written to {directory / f'metrics_{args.split}.json'}")


def _cmd_predict(config: ProjectConfig, args: argparse.Namespace) -> None:
    from cuneiform.inference import Predictor

    predictor = Predictor.from_latest(args.config, run=args.run, device=config.training.device)
    result = predictor.predict_scan(args.ply) if args.ply else predictor.predict_tablet(args.tablet_id)
    print(json.dumps(result.to_dict(), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cuneiform", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH), help="path to the YAML config")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("fetch-metadata", help="download HeiCuBeDa metadata and index the scan archives (~15 MB)")
    sub.add_parser("build-manifest", help="sample labelled HeiCuBeDa tablets and assign tablet-level splits")
    sub.add_parser("assign-splits", help="fill empty 'split' cells of a hand-written manifest (by tablet)")
    sub.add_parser("summarize", help="class counts per split and depth-map cache status")

    prepare = sub.add_parser("prepare", help="download missing scans and cache depth maps (resumable)")
    prepare.add_argument("--workers", type=int, default=3, help="parallel downloads")
    prepare.add_argument("--limit", type=int, help="process at most N new tablets")
    prepare.add_argument("--tablet", action="append", help="only this tablet ID (repeatable)")
    prepare.add_argument("--retry-invalid", action="store_true", help="retry scans logged as invalid")
    prepare.add_argument("--keep-raw", action="store_true", help="keep downloaded .ply files after caching")

    preview = sub.add_parser("preview", help="save a PNG of a tablet's depth maps")
    target = preview.add_mutually_exclusive_group(required=True)
    target.add_argument("tablet_id", nargs="?")
    target.add_argument("--ply", help="render a .ply file directly")

    train = sub.add_parser("train", help="fine-tune ResNet18 and save a model bundle")
    train.add_argument("--epochs", type=int, help="override training.epochs")
    train.add_argument("--run-name", help="bundle directory name (default: timestamp)")
    train.add_argument("--no-evaluate", action="store_true", help="skip the validation report")

    evaluate = sub.add_parser("evaluate", help="evaluate a bundle on the test split (final numbers)")
    evaluate.add_argument("--run", help="bundle name or path (default: artifacts/models/LATEST)")
    evaluate.add_argument("--split", default="test", choices=["train", "val", "test"])

    predict = sub.add_parser("predict", help="predict one tablet and print JSON")
    target = predict.add_mutually_exclusive_group(required=True)
    target.add_argument("tablet_id", nargs="?")
    target.add_argument("--ply")
    predict.add_argument("--run")
    return parser


COMMANDS = {
    "fetch-metadata": _cmd_fetch_metadata,
    "build-manifest": _cmd_build_manifest,
    "assign-splits": _cmd_assign_splits,
    "summarize": _cmd_summarize,
    "prepare": _cmd_prepare,
    "preview": _cmd_preview,
    "train": _cmd_train,
    "evaluate": _cmd_evaluate,
    "predict": _cmd_predict,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s", datefmt="%H:%M:%S")
    try:
        config = load_config(args.config)
        COMMANDS[args.command](config, args)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
