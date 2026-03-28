import argparse
from pathlib import Path
from typing import Iterable, List, Set

import h5py


def read_list_file(list_path: Path) -> List[Path]:
    files: List[Path] = []
    for line in list_path.read_text().splitlines():
        item = line.strip()
        if not item or item.startswith("#"):
            continue
        files.append(Path(item))
    return files


def collect_files(list_files: Iterable[Path], direct_files: Iterable[Path]) -> List[Path]:
    seen: Set[Path] = set()
    result: List[Path] = []

    for p in direct_files:
        rp = p.resolve()
        if rp not in seen:
            seen.add(rp)
            result.append(rp)

    for lf in list_files:
        for p in read_list_file(lf):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                result.append(rp)

    return result


def update_source(files: Iterable[Path], source_value: str, dry_run: bool = False) -> None:
    total = 0
    changed = 0
    missing = 0

    for fpath in files:
        total += 1
        if not fpath.exists():
            missing += 1
            print(f"[MISSING] {fpath}")
            continue

        old_val = None
        with h5py.File(fpath, "a") as h5f:
            old_val = h5f.attrs.get("source", None)
            if not dry_run:
                h5f.attrs["source"] = source_value

        if str(old_val) != source_value:
            changed += 1
        action = "DRY-RUN" if dry_run else "UPDATED"
        print(f"[{action}] {fpath} | source: {old_val} -> {source_value}")

    print("\nSummary")
    print(f"- total files : {total}")
    print(f"- changed     : {changed}")
    print(f"- missing     : {missing}")
    print(f"- dry_run     : {dry_run}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch set HDF5 root attribute 'source'."
    )
    parser.add_argument(
        "--source",
        required=True,
        help="New source value, e.g. hqfd",
    )
    parser.add_argument(
        "--list",
        dest="list_files",
        nargs="*",
        default=["datasets/HQF/train_list.txt", "datasets/HQF/val_list.txt"],
        help="Text files containing HDF5 paths (one per line)",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[],
        help="Direct HDF5 file paths",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview changes without writing",
    )

    args = parser.parse_args()

    list_files = [Path(p) for p in args.list_files]
    missing_lists = [p for p in list_files if not p.exists()]
    if missing_lists:
        print("List files not found:")
        for p in missing_lists:
            print(f"- {p}")
        raise SystemExit(1)

    direct_files = [Path(p) for p in args.files]
    files = collect_files(list_files=list_files, direct_files=direct_files)
    if not files:
        print("No input HDF5 files found.")
        return

    update_source(files, source_value=args.source, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
