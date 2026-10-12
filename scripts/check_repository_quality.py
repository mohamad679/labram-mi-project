"""Offline repository quality gates: local Markdown links and tracked sensitive artifacts.

Intentionally does not check external URLs (network changes) or assert source
license compliance. Run from repository root:
    python -m scripts.check_repository_quality
"""
from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path
from urllib.parse import unquote, urlsplit

LINK_RE = re.compile(r"!?\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))(?:\s+[^)]*)?\)")
FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
DISALLOWED_EXTENSIONS = {
    ".npz", ".npy", ".pt", ".pth", ".ckpt", ".edf", ".bdf",
    ".fif", ".set", ".pkl", ".pickle", ".joblib",
}
DISALLOWED_NAMES = {
    ".env", ".env.local", ".env.production", "id_rsa", "id_ed25519",
    "credentials.json", "service-account.json",
}


def tracked_paths(root: Path) -> list[Path]:
    """Read the git index to avoid scanning gitignored local datasets."""
    result = subprocess.run(
        ["git", "-C", str(root), "ls-files", "-z"],
        capture_output=True,
        check=True,
    )
    return [Path(item.decode("utf-8")) for item in result.stdout.split(b"\x00") if item]


def markdown_targets(markdown: str) -> list[str]:
    """Extract inline links while ignoring fenced code examples."""
    targets = []
    fenced = False
    fence_char = ""
    fence_length = 0
    for line in markdown.splitlines():
        marker = FENCE_RE.match(line)
        if marker:
            found = marker.group(1)
            if not fenced:
                fenced, fence_char, fence_length = True, found[0], len(found)
            elif found[0] == fence_char and len(found) >= fence_length:
                fenced = False
            continue
        if fenced:
            continue
        for match in LINK_RE.finditer(line):
            targets.append(match.group(1) or match.group(2))
    return targets


def check_markdown_links(root: Path, paths: list[Path]) -> list[str]:
    errors = []
    root = root.resolve()
    for source in paths:
        if source.suffix.lower() != ".md" or not (root / source).is_file():
            continue
        for target in markdown_targets((root / source).read_text(encoding="utf-8")):
            if target.startswith(("#", "mailto:", "data:")):
                continue
            parsed = urlsplit(target)
            if parsed.scheme or parsed.netloc or target.startswith("//"):
                continue
            relative = unquote(parsed.path)
            if not relative:
                continue
            if relative.startswith("/"):
                errors.append(f"{source}: absolute link {target!r}; use relative repo path")
                continue
            resolved = (root / source.parent / relative).resolve()
            if not resolved.is_relative_to(root):
                errors.append(f"{source}: link escapes repository: {target!r}")
            elif not resolved.is_file():
                errors.append(f"{source}: missing link target: {target!r}")
    return errors


def check_tracked_artifacts(paths: list[Path]) -> list[str]:
    errors = []
    for path in paths:
        lowered = path.name.lower()
        if path.suffix.lower() in DISALLOWED_EXTENSIONS or lowered in DISALLOWED_NAMES:
            errors.append(f"Disallowed tracked data/credential artifact: {path}")
    return errors


def audit_repository(root: Path) -> list[str]:
    paths = tracked_paths(root)
    return check_tracked_artifacts(paths) + check_markdown_links(root, paths)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", default=".", type=Path)
    args = parser.parse_args()
    failures = audit_repository(args.repo_root)
    if failures:
        print("Repository quality checks FAILED:")
        for failure in failures:
            print(" -", failure)
        return 1
    print("Repository quality checks passed: tracked artifact policy and internal Markdown links.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
