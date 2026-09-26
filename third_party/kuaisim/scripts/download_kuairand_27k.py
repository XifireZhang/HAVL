#!/usr/bin/env python3
"""Download the official KuaiRand-27K archive with resumable connections."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import urllib.request
from pathlib import Path


API_URL = "https://recapi.ustc.edu.cn/api/v2/share/download"
SHARE_ID = "7f5a44c0-c196-11ed-b626-81cd6a642112"
RESOURCE_ID = "b2210240-e6ee-11ec-a0bf-e75afe6668d1"
ARCHIVE_NAME = "KuaiRand-27K.tar.gz"
EXPECTED_MD5 = "3e3c799a24e2d23a4d2c757fbf9adf59"


def get_download_url() -> str:
    payload = {
        "share_number": SHARE_ID,
        "share_constraint": {"password": None},
        "share_resources_list": [RESOURCE_ID],
    }
    request = urllib.request.Request(
        API_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.loads(response.read().decode("utf-8-sig"))
    if result.get("status_code") != 200:
        raise RuntimeError(f"Download API failed: {result.get('message', result)}")
    return result["entity"][RESOURCE_ID]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--connections", type=int, default=16)
    args = parser.parse_args()

    aria2c = shutil.which("aria2c")
    if aria2c is None:
        raise RuntimeError("aria2c is required for resumable downloads")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    url = get_download_url()
    command = [
        aria2c,
        "--continue=true",
        f"--max-connection-per-server={args.connections}",
        f"--split={args.connections}",
        "--min-split-size=16M",
        "--file-allocation=none",
        f"--dir={args.output_dir}",
        f"--out={ARCHIVE_NAME}",
        url,
    ]
    print(f"Downloading {ARCHIVE_NAME} to {args.output_dir}")
    print(f"Expected MD5 after completion: {EXPECTED_MD5}")
    os.execv(aria2c, command)


if __name__ == "__main__":
    main()
