#!/usr/bin/env python3
"""Extract and preprocess the 27K feature files required by KuaiSim."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tarfile
from pathlib import Path, PurePosixPath

import pandas as pd


EXPECTED_MD5 = "3e3c799a24e2d23a4d2c757fbf9adf59"
RAW_USER_FILE = "user_features_27k.csv"
RAW_VIDEO_FILE = "video_features_basic_27k.csv"
OUTPUT_USER_FILE = "user_features_27K_fillna.csv"
OUTPUT_VIDEO_FILE = "video_features_basic_27K_fillna.csv"


def md5sum(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def extract_required_files(archive_path: Path, output_dir: Path) -> None:
    wanted = {RAW_USER_FILE, RAW_VIDEO_FILE}
    extracted: set[str] = set()
    with tarfile.open(archive_path, mode="r|gz") as archive:
        for member in archive:
            path = PurePosixPath(member.name)
            if not member.isfile() or path.name not in wanted:
                continue
            if len(path.parts) < 2 or path.parts[-2] != "data":
                continue
            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Cannot read {member.name} from archive")
            destination = output_dir / path.name
            with source, destination.open("wb") as target:
                shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
            extracted.add(path.name)
            print(f"Extracted {member.name} -> {destination}")
            if extracted == wanted:
                break
    missing = wanted - extracted
    if missing:
        raise RuntimeError(f"Archive is missing required files: {sorted(missing)}")


def preprocess_user_features(output_dir: Path) -> None:
    source = output_dir / RAW_USER_FILE
    destination = output_dir / OUTPUT_USER_FILE
    frame = pd.read_csv(source)
    if "user_id" not in frame or frame["user_id"].duplicated().any():
        raise RuntimeError("Invalid user feature table: missing or duplicate user_id")
    frame.fillna(0).to_csv(destination, index=False)
    print(f"Prepared {destination} ({len(frame):,} users)")


def preprocess_video_features(output_dir: Path, chunk_size: int) -> None:
    source = output_dir / RAW_VIDEO_FILE
    destination = output_dir / OUTPUT_VIDEO_FILE
    first_chunk = True
    row_count = 0
    for frame in pd.read_csv(source, chunksize=chunk_size):
        required = {"video_id", "tag", "music_type"}
        missing = required - set(frame.columns)
        if missing:
            raise RuntimeError(f"Video feature table is missing columns: {sorted(missing)}")
        frame["tag"] = frame["tag"].fillna(0)
        frame["music_type"] = frame["music_type"].fillna(0)
        frame.to_csv(
            destination,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
        )
        first_chunk = False
        row_count += len(frame)
        print(f"Prepared {row_count:,} video rows", flush=True)
    print(f"Prepared {destination} ({row_count:,} videos)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--chunk-size", type=int, default=500_000)
    parser.add_argument("--skip-md5", action="store_true")
    args = parser.parse_args()

    if not args.archive.is_file():
        raise FileNotFoundError(args.archive)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_md5:
        actual_md5 = md5sum(args.archive)
        if actual_md5 != EXPECTED_MD5:
            raise RuntimeError(
                f"MD5 mismatch: expected {EXPECTED_MD5}, found {actual_md5}"
            )
        print(f"Archive MD5 verified: {actual_md5}")

    extract_required_files(args.archive, args.output_dir)
    preprocess_user_features(args.output_dir)
    preprocess_video_features(args.output_dir, args.chunk_size)


if __name__ == "__main__":
    main()
