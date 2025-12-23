import os
from pathlib import Path


def count_lines(directory: str, recursive: bool = True) -> dict[str, int]:
    """
    Count lines of all .py files in a directory.
    
    Args:
        directory: Path to the directory to scan
        recursive: If True, scan subdirectories as well
        
    Returns:
        Dictionary mapping file paths to their line counts
    """
    directory = Path(directory)
    results = {}
    
    pattern = "**/*.py" if recursive else "*.py"
    
    for py_file in directory.glob(pattern):
        if py_file.is_file():
            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    line_count = sum(1 for _ in f)
                results[str(py_file)] = line_count
            except (IOError, UnicodeDecodeError):
                # Skip files that can't be read
                continue
    
    return results


def print_summary(line_counts: dict[str, int]) -> None:
    """Print a formatted summary of line counts."""
    if not line_counts:
        print("No .py files found.")
        return
    
    # Sort by line count descending
    sorted_counts = sorted(line_counts.items(), key=lambda x: x[1], reverse=True)
    
    print(f"{'File':<60} {'Lines':>8}")
    print("-" * 70)
    
    for filepath, count in sorted_counts:
        # Truncate long paths
        display_path = filepath if len(filepath) <= 58 else "..." + filepath[-55:]
        print(f"{display_path:<60} {count:>8}")
    
    print("-" * 70)
    print(f"{'Total':<60} {sum(line_counts.values()):>8}")
    print(f"Files: {len(line_counts)}")


if __name__ == "__main__":
    import sys
    
    target_dir = sys.argv[1] if len(sys.argv) > 1 else "."
    counts = count_lines(target_dir)
    print_summary(counts)

