"""Pack the sweep outputs into ~1 GB tar shards and upload them to the onset dataset repo, then verify sizes."""

from __future__ import annotations

import os
import sys
import tarfile
from pathlib import Path

from huggingface_hub import HfApi

REPO = "gavinlaw/sglang-omni-qwen3-tts-base-onset-en-audio"
SHARD_BYTES = 1 << 30


def main() -> None:
    run_dir, prefix = Path(sys.argv[1]), sys.argv[2]
    stage = run_dir / "hf_upload"
    stage.mkdir(exist_ok=True)
    jobs = sorted(p for p in (run_dir / "out").iterdir() if p.is_dir())
    shards: list[list[Path]] = [[]]
    size = 0
    for job in jobs:
        job_bytes = sum(f.stat().st_size for f in job.rglob("*") if f.is_file())
        if shards[-1] and size + job_bytes > SHARD_BYTES:
            shards.append([])
            size = 0
        shards[-1].append(job)
        size += job_bytes
    for index, members in enumerate(shards):
        with tarfile.open(stage / f"out-{index:02d}.tar", "w") as archive:
            for job in members:
                archive.add(job, arcname=f"out/{job.name}")
    api = HfApi(token=os.environ["HF_TOKEN"])
    print("whoami", api.whoami()["name"])
    commit = api.upload_folder(folder_path=stage, repo_id=REPO, repo_type="dataset", path_in_repo=prefix,
                               commit_message=f"Mask-length sweep {prefix}")
    info = {s.rfilename: s.size for s in api.dataset_info(REPO, revision=commit.oid, files_metadata=True).siblings}
    local = {f"{prefix}/{f.name}": f.stat().st_size for f in stage.iterdir()}
    mismatched = {k: (info.get(k), v) for k, v in local.items() if info.get(k) != v}
    print("commit", commit.oid, "shards", len(local), "jobs", len(jobs), "mismatched", mismatched)


if __name__ == "__main__":
    main()
