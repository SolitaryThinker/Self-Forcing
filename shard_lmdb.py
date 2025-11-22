#!/usr/bin/env python3
"""Utilities for sharding and merging LMDB datasets.

This script can:
1. Deterministically shard an LMDB environment into multiple LMDB shards.
2. Merge several shard LMDB environments back into a single LMDB.
3. Compare two LMDB environments to verify they contain identical entries.

Example usages:
    # Split into shards of 50_000 rows each
    python shard_lmdb.py shard \
        --input /data/full_dataset \
        --output-dir /data/shards \
        --rows-per-shard 50000

    # Merge shards back together, overwriting duplicate keys
    python shard_lmdb.py merge \
        --shards-dir /data/shards \
        --output /data/full_dataset_merged \
        --on-duplicate overwrite
"""
from __future__ import annotations

import argparse
import math
import os
import shutil
import sys
from pathlib import Path
from typing import Iterable, List, Sequence

import lmdb  # type: ignore[import]


DEFAULT_MAP_SIZE = 400 * 1024 ** 3  # 16 GiB
DEFAULT_TXN_SIZE = 1024


def positive_int(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"{value!r} is not an integer") from exc
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def log_message(level: str, message: str, *args: object) -> None:
    formatted = message % args if args else message
    print(f"[{level.upper()}] {formatted}")


def log_info(message: str, *args: object) -> None:
    log_message("INFO", message, *args)


def log_warning(message: str, *args: object) -> None:
    log_message("WARNING", message, *args)


def log_error(message: str, *args: object) -> None:
    log_message("ERROR", message, *args)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Shard or merge LMDB datasets deterministically.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log-every",
        type=positive_int,
        default=10_000,
        help="log progress every N rows processed",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    shard_parser = subparsers.add_parser(
        "shard",
        help="split an LMDB environment into deterministic shards",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    shard_parser.add_argument("--input", required=True, help="path to source LMDB directory")
    shard_parser.add_argument(
        "--output-dir",
        required=True,
        help="directory where shard LMDBs will be written (will be created or emptied)",
    )
    shard_parser.add_argument(
        "--shard-prefix",
        default="shard",
        help="prefix for shard directory names",
    )
    shard_parser.add_argument(
        "--rows-per-shard",
        type=positive_int,
        help="maximum number of rows per shard (mutually exclusive with --num-shards)",
    )
    shard_parser.add_argument(
        "--num-shards",
        type=positive_int,
        help="target number of shards; overrides --rows-per-shard",
    )
    shard_parser.add_argument(
        "--map-size",
        type=positive_int,
        default=DEFAULT_MAP_SIZE,
        help="map size (bytes) to allocate for each shard LMDB",
    )
    shard_parser.add_argument(
        "--txn-size",
        type=positive_int,
        default=DEFAULT_TXN_SIZE,
        help="number of rows to buffer per write transaction",
    )
    shard_parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="allow non-empty output directory by wiping it first",
    )

    merge_parser = subparsers.add_parser(
        "merge",
        help="merge shard LMDBs back into a single LMDB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    merge_parser.add_argument(
        "--shards-dir",
        help="directory containing shard LMDB subdirectories",
    )
    merge_parser.add_argument(
        "--shard",
        dest="shard_paths",
        action="append",
        default=[],
        help="path to an individual shard (can be used multiple times)",
    )
    merge_parser.add_argument("--output", required=True, help="path for merged LMDB directory")
    merge_parser.add_argument(
        "--map-size",
        type=positive_int,
        default=DEFAULT_MAP_SIZE,
        help="map size (bytes) for the merged LMDB",
    )
    merge_parser.add_argument(
        "--txn-size",
        type=positive_int,
        default=DEFAULT_TXN_SIZE,
        help="number of rows to buffer per write transaction",
    )
    merge_parser.add_argument(
        "--on-duplicate",
        choices=("error", "skip", "overwrite"),
        default="error",
        help="behavior when duplicate keys are encountered during merge",
    )
    merge_parser.add_argument(
        "--overwrite-output",
        action="store_true",
        help="allow non-empty output directory by wiping it first",
    )

    compare_parser = subparsers.add_parser(
        "compare",
        help="compare two LMDB environments for key/value equality",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    compare_parser.add_argument(
        "--first",
        required=True,
        help="path to the first LMDB directory",
    )
    compare_parser.add_argument(
        "--second",
        required=True,
        help="path to the second LMDB directory",
    )

    return parser.parse_args()


def resolve_path(path: str) -> Path:
    return Path(path).expanduser().resolve()


def ensure_empty_dir(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not path.is_dir():
            raise ValueError(f"{path} exists and is not a directory")
        contents = list(path.iterdir())
        if contents:
            if not overwrite:
                raise ValueError(f"{path} is not empty; use --overwrite-output to remove it")
            log_info("Removing existing directory %s", path)
            shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def is_lmdb_dir(path: Path) -> bool:
    return path.is_dir() and (path / "data.mdb").exists()


def summarize_bytes(blob: bytes, limit: int = 32) -> str:
    if not blob:
        return "<empty>"
    snippet = blob[:limit].hex()
    if len(blob) > limit:
        return f"{snippet}... (len={len(blob)})"
    return f"{snippet} (len={len(blob)})"


def open_lmdb_env(path: Path, *, readonly: bool, map_size: int | None = None) -> lmdb.Environment:
    kwargs = {
        "path": os.fspath(path),
        "subdir": True,
        "readonly": readonly,
        "lock": not readonly,
        "readahead": readonly,
        "create": not readonly,
        "max_readers": 512,
        "meminit": False,
    }
    if map_size is not None:
        kwargs["map_size"] = map_size
    return lmdb.open(**kwargs)


def count_entries(env: lmdb.Environment) -> int:
    with env.begin(write=False) as txn:
        stats = txn.stat()
    return stats.get("entries", 0)


def shard_lmdb(args: argparse.Namespace) -> None:
    source = resolve_path(args.input)
    output_dir = resolve_path(args.output_dir)
    if source == output_dir:
        raise ValueError("Input and output paths must differ")

    ensure_empty_dir(output_dir, overwrite=args.overwrite_output)

    log_info("Opening source LMDB at %s", source)
    src_env = open_lmdb_env(source, readonly=True)
    total_entries = count_entries(src_env)
    if total_entries == 0:
        log_warning("Source LMDB contains no entries; nothing to shard")
        return

    if args.num_shards:
        rows_per_shard = math.ceil(total_entries / args.num_shards)
        log_info(
            "Total entries: %d | Target shards: %d | Rows per shard: %d",
            total_entries,
            args.num_shards,
            rows_per_shard,
        )
    else:
        if not args.rows_per_shard:
            raise ValueError("Either --rows-per-shard or --num-shards must be provided")
        rows_per_shard = args.rows_per_shard
        log_info(
            "Total entries: %d | Rows per shard: %d (expected shards ≈ %d)",
            total_entries,
            rows_per_shard,
            math.ceil(total_entries / rows_per_shard),
        )

    shard_index = -1
    rows_in_current = 0
    rows_written = 0
    shard_env: lmdb.Environment | None = None
    write_txn: lmdb.Transaction | None = None

    def start_new_shard() -> lmdb.Environment:
        nonlocal shard_index, rows_in_current
        shard_index += 1
        shard_name = f"{args.shard_prefix}_{shard_index:05d}"
        shard_path = output_dir / shard_name
        shard_path.mkdir(parents=True, exist_ok=True)
        log_info("Starting shard %s", shard_path)
        rows_in_current = 0
        return open_lmdb_env(shard_path, readonly=False, map_size=args.map_size)

    try:
        shard_env = start_new_shard()
        with src_env.begin(write=False) as read_txn:
            cursor = read_txn.cursor()
            for key, value in cursor:
                if rows_in_current >= rows_per_shard:
                    if write_txn is not None:
                        write_txn.commit()
                        write_txn = None
                    assert shard_env is not None
                    shard_env = start_new_shard()
                assert shard_env is not None
                if write_txn is None:
                    write_txn = shard_env.begin(write=True)
                txn = write_txn
                assert txn is not None
                txn.put(key, value, overwrite=True)
                rows_in_current += 1
                rows_written += 1
                if rows_written % args.log_every == 0:
                    log_info(
                        "Sharding progress: %d rows written (%d shard(s) started)",
                        rows_written,
                        shard_index + 1,
                    )
                if rows_written % args.txn_size == 0:
                    txn.commit()
                    write_txn = None

        if write_txn is not None:
            write_txn.commit()
            write_txn = None
        log_info(
            "Completed sharding: %d total rows across %d shard(s)",
            rows_written,
            shard_index + 1,
        )
    finally:
        if write_txn is not None:
            write_txn.abort()
        if shard_env is not None:
            shard_env.close()
        src_env.close()


def collect_shard_paths(shards_dir: str | None, explicit_paths: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    seen = set()

    for raw in explicit_paths:
        path = resolve_path(raw)
        if path in seen:
            continue
        if not is_lmdb_dir(path):
            raise ValueError(f"{path} is not a valid LMDB directory")
        paths.append(path)
        seen.add(path)

    if shards_dir:
        base = resolve_path(shards_dir)
        if not base.exists():
            raise ValueError(f"{base} does not exist")
        if not base.is_dir():
            raise ValueError(f"{base} is not a directory")
        for child in sorted(base.iterdir()):
            if child in seen:
                continue
            if is_lmdb_dir(child):
                paths.append(child)
                seen.add(child)

    return paths


def merge_lmdb(args: argparse.Namespace) -> None:
    shard_paths = collect_shard_paths(args.shards_dir, args.shard_paths)
    if not shard_paths:
        raise ValueError("No shard directories provided or discovered")

    output_path = resolve_path(args.output)
    ensure_empty_dir(output_path, overwrite=args.overwrite_output)

    log_info("Merging %d shard(s) into %s", len(shard_paths), output_path)
    target_env = open_lmdb_env(output_path, readonly=False, map_size=args.map_size)
    total_written = 0

    try:
        for shard in shard_paths:
            log_info("Reading shard %s", shard)
            shard_env = open_lmdb_env(shard, readonly=True)
            try:
                with shard_env.begin(write=False) as read_txn:
                    cursor = read_txn.cursor()
                    write_txn: lmdb.Transaction | None = None
                    buffered = 0
                    for key, value in cursor:
                        if write_txn is None:
                            write_txn = target_env.begin(write=True)
                            buffered = 0
                        txn = write_txn
                        assert txn is not None
                        if args.on_duplicate == "error":
                            success = txn.put(key, value, overwrite=False)
                            if not success:
                                txn.abort()
                                write_txn = None
                                raise ValueError(
                                    f"Duplicate key encountered: {key!r} in shard {shard}"
                                )
                        elif args.on_duplicate == "skip":
                            txn.put(key, value, overwrite=False)
                        else:  # overwrite
                            txn.put(key, value, overwrite=True)
                        total_written += 1
                        buffered += 1
                        if total_written % args.log_every == 0:
                            log_info(
                                "Merging progress: %d rows written from %s",
                                total_written,
                                shard,
                            )
                        if buffered >= args.txn_size:
                            txn.commit()
                            write_txn = None
                    if write_txn is not None:
                        write_txn.commit()
            finally:
                shard_env.close()

        log_info("Merge complete. Total rows written: %d", total_written)
    finally:
        target_env.close()


def compare_lmdb(args: argparse.Namespace) -> None:
    first = resolve_path(args.first)
    second = resolve_path(args.second)
    if first == second:
        raise ValueError("Comparison targets must be different directories")
    if not is_lmdb_dir(first):
        raise ValueError(f"{first} is not a valid LMDB directory")
    if not is_lmdb_dir(second):
        raise ValueError(f"{second} is not a valid LMDB directory")

    log_info("Comparing LMDBs: %s ↔ %s", first, second)
    env_a = open_lmdb_env(first, readonly=True)
    env_b = open_lmdb_env(second, readonly=True)

    try:
        entries_a = count_entries(env_a)
        entries_b = count_entries(env_b)
        log_info("Entry counts — first: %d | second: %d", entries_a, entries_b)
        if entries_a != entries_b:
            raise ValueError(
                f"Entry count mismatch: first={entries_a} second={entries_b}"
            )

        compared = 0
        with env_a.begin(write=False) as txn_a, env_b.begin(write=False) as txn_b:
            cursor_a = txn_a.cursor()
            cursor_b = txn_b.cursor()
            valid_a = cursor_a.first()
            valid_b = cursor_b.first()

            while valid_a and valid_b:
                key_a = cursor_a.key()
                key_b = cursor_b.key()
                if key_a != key_b:
                    raise ValueError(
                        "Key mismatch at entry %d: first=%s second=%s"
                        % (compared, summarize_bytes(key_a), summarize_bytes(key_b))
                    )
                value_a = cursor_a.value()
                value_b = cursor_b.value()
                if value_a != value_b:
                    raise ValueError(
                        "Value mismatch for key %s: first=%s second=%s"
                        % (
                            summarize_bytes(key_a),
                            summarize_bytes(value_a),
                            summarize_bytes(value_b),
                        )
                    )
                compared += 1
                if compared % args.log_every == 0:
                    log_info("Compare progress: %d entries validated", compared)
                valid_a = cursor_a.next()
                valid_b = cursor_b.next()

            if valid_a or valid_b:
                raise ValueError(
                    "Data length mismatch detected after %d entries" % compared
                )

        log_info("Comparison successful: %d entries identical", compared)
    finally:
        env_a.close()
        env_b.close()


def main() -> None:
    args = parse_args()

    if args.command == "shard":
        shard_lmdb(args)
    elif args.command == "merge":
        merge_lmdb(args)
    elif args.command == "compare":
        compare_lmdb(args)
    else:
        raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - CLI guardrail
        log_error(str(exc))
        sys.exit(1)

