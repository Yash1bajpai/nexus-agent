"""No secret values are read by this gate. Output only fixed classifications."""
import argparse
from pathlib import Path
import re
import shutil


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("repo", nargs="?", type=Path)
    parser.add_argument("--scan", type=Path)
    args = parser.parse_args()
    if args.scan is not None:
        root = args.scan
        if not root.is_dir():
            return 1
        for path in root.rglob("*"):
            if path.is_symlink() or (path.is_file() and re.search(rb"sk-or-[A-Za-z0-9_-]+", path.read_bytes())):
                shutil.rmtree(root)
                print("ARTIFACT_SCAN_FAIL_DELETED")
                return 2
        print("ARTIFACT_SCAN_PASS")
        return 0
    if args.repo is None:
        return 1
    references = [p.relative_to(args.repo).as_posix() for p in
                  (args.repo / ".github/workflows").iterdir()
                  if p.suffix in (".yml", ".yaml") and "OPENROUTER_API_KEY" in p.read_text()]
    if references != [".github/workflows/bench.yml"]:
        print("SECRET_REFERENCE_SCOPE_FAIL")
        return 1
    print("SECRET_REFERENCE_SCOPE_PASS")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
