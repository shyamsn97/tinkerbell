"""Count physical source lines in the Tinkerbell library."""

from __future__ import annotations

import argparse
from pathlib import Path

DEFAULT_EXCLUDES = {"__pycache__", ".mypy_cache", ".pytest_cache", ".ruff_cache"}


def count_lines(directory: Path, recursive: bool = True) -> dict[Path, int]:
    """Count physical lines in all Python files under a directory."""
    results: dict[Path, int] = {}
    pattern = "**/*.py" if recursive else "*.py"
    for py_file in directory.glob(pattern):
        if not py_file.is_file():
            continue
        if any(part in DEFAULT_EXCLUDES for part in py_file.parts):
            continue
        try:
            with py_file.open("r", encoding="utf-8") as f:
                results[py_file] = sum(1 for _ in f)
        except UnicodeDecodeError:
            continue
    return results


def print_summary(line_counts: dict[Path, int], root: Path) -> None:
    """Print a formatted summary of line counts."""
    if not line_counts:
        print("No .py files found.")
        return

    sorted_counts = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)

    print(f"{'File':<60} {'Lines':>8}")
    print("-" * 70)

    for filepath, count in sorted_counts:
        try:
            display_path = str(filepath.relative_to(root))
        except ValueError:
            display_path = str(filepath)
        if len(display_path) > 58:
            display_path = "..." + display_path[-55:]
        print(f"{display_path:<60} {count:>8}")

    print("-" * 70)
    print(f"{'Total':<60} {sum(line_counts.values()):>8}")
    print(f"Files: {len(line_counts)}")


def parse_args() -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Count physical Python source lines in the Tinkerbell library."
    )
    parser.add_argument(
        "path",
        nargs="?",
        default=repo_root / "tinkerbell",
        type=Path,
        help="Directory to count. Defaults to ./tinkerbell.",
    )
    parser.add_argument(
        "--non-recursive",
        action="store_true",
        help="Only count Python files directly inside the target directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target = args.path.resolve()
    counts = count_lines(target, recursive=not args.non_recursive)
    print_summary(counts, root=target)


if __name__ == "__main__":
    main()

