"""Publish Heliograph to a Hugging Face Space (Docker SDK, free CPU hardware).

    pip install huggingface_hub
    hf auth login                                   # paste a token with "write" permission
    python deploy/huggingface/publish.py --space <your-username>/heliograph
    python deploy/huggingface/publish.py --space <your-username>/heliograph --dry-run   # only show what would be sent

Why a script instead of ``git push``: the repository ships about 800 binary files (trained ``.pt`` checkpoints,
``.npz`` arrays, ``.pkl`` models, ``.png`` screenshots). The Hugging Face Hub rejects binary files in a plain
``git push`` unless they are stored with Git LFS / Xet. ``HfApi.upload_folder`` handles that automatically.

The script copies the files tracked by git (or, without git, the project folder minus local environments and
data) into a temporary folder, puts ``deploy/huggingface/Dockerfile`` and ``deploy/huggingface/README.md``
(the Space card) at its root, creates the Space if it does not exist, and uploads everything in one commit.
The Space then builds the image itself (about 10-15 minutes: PyTorch, the UI build and the NASA dataset).
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SKIP_DIRS = {".git", "node_modules", ".venv", "data", "artifacts", "logs", ".run", "dist", "__pycache__",
             ".pytest_cache", "test-results", "playwright-report"}


def project_files() -> list[Path]:
    """Relative paths of the files to publish: git-tracked files, or a filtered walk without git."""
    try:
        out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
        files = [Path(p) for p in out.stdout.decode().split("\0") if p]
        if files:
            return [f for f in files if (ROOT / f).is_file()]
    except (OSError, subprocess.CalledProcessError):
        pass
    files = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for name in filenames:
            p = Path(dirpath, name).relative_to(ROOT)
            if p.suffix not in {".pyc", ".log"} and not name.endswith((".db", ".db-wal", ".db-shm")):
                files.append(p)
    return files


def stage(dest: Path) -> int:
    count = 0
    for rel in project_files():
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, target)
        count += 1
    # The Space needs its Dockerfile and its card (README.md with the YAML header) at the repository root.
    shutil.copy2(HERE / "Dockerfile", dest / "Dockerfile")
    shutil.copy2(HERE / "README.md", dest / "README.md")
    return count


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--space", required=True, help="Space id, e.g. your-username/heliograph")
    ap.add_argument("--token", default=None, help="Hugging Face write token (default: HF_TOKEN or `hf auth login`)")
    ap.add_argument("--private", action="store_true", help="create the Space as private")
    ap.add_argument("--dry-run", action="store_true", help="stage the files and print a summary; upload nothing")
    args = ap.parse_args()

    if "/" not in args.space:
        print("--space must look like <your-username>/<space-name>", file=sys.stderr)
        return 2

    with tempfile.TemporaryDirectory(prefix="heliograph-space-") as tmp:
        dest = Path(tmp)
        n = stage(dest)
        size_mb = sum(p.stat().st_size for p in dest.rglob("*") if p.is_file()) / 1e6
        print(f"Staged {n} project files ({size_mb:.1f} MB) plus the Space Dockerfile and card.")
        if args.dry_run:
            for name in ("Dockerfile", "README.md", "backend/requirements.txt", "reference/manifest.json"):
                print(f"  {'ok ' if (dest / name).exists() else 'MISSING'} {name}")
            print("Dry run: nothing uploaded.")
            return 0

        try:
            from huggingface_hub import HfApi
        except ImportError:
            print("huggingface_hub is not installed. Run:  pip install huggingface_hub", file=sys.stderr)
            return 1

        api = HfApi(token=args.token)
        try:
            user = api.whoami()["name"]
        except Exception as exc:  # noqa: BLE001
            print(f"Not logged in to Hugging Face ({exc}).\nRun `hf auth login` (or set HF_TOKEN) with a token "
                  "that has write permission, then try again.", file=sys.stderr)
            return 1
        print(f"Logged in as {user}. Creating/updating Space {args.space} ...")
        api.create_repo(args.space, repo_type="space", space_sdk="docker", private=args.private, exist_ok=True)
        api.upload_folder(folder_path=str(dest), repo_id=args.space, repo_type="space",
                          commit_message="Publish Heliograph")
    print(f"Uploaded. The Space is building now: https://huggingface.co/spaces/{args.space}")
    print("The first build takes about 10-15 minutes. Open the link and watch the 'Logs' tab.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
