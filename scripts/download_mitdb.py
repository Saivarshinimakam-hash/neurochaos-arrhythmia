"""
scripts/download_mitdb.py
--------------------------
Download all 48 MIT-BIH Arrhythmia Database records from PhysioNet.

- Skips records already complete (.dat + .hea + .atr all present).
- Removes partial downloads before re-attempting (avoids corrupt files).
- Prints a final audit table of pass / fail.

Run from project root:
    python scripts/download_mitdb.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import wfdb  # type: ignore

DEST = PROJECT_ROOT / "data" / "raw" / "mitdb"
DEST.mkdir(parents=True, exist_ok=True)

ALL_RECORDS = [
    100, 101, 102, 103, 104, 105, 106, 107, 108, 109,
    111, 112, 113, 114, 115, 116, 117, 118, 119, 121,
    122, 123, 124, 200, 201, 202, 203, 205, 207, 208,
    209, 210, 212, 213, 214, 215, 217, 219, 220, 221,
    222, 223, 228, 230, 231, 232, 233, 234,
]


def is_complete(record_id: int) -> bool:
    base = DEST / str(record_id)
    return (base.with_suffix(".dat").exists() and
            base.with_suffix(".hea").exists() and
            base.with_suffix(".atr").exists())


def remove_partial(record_id: int) -> None:
    """Remove any partially-downloaded files for a record."""
    for ext in (".dat", ".hea", ".atr"):
        p = DEST / (str(record_id) + ext)
        if p.exists():
            p.unlink()
            print(f"  Removed partial file: {p.name}")


def download_record(record_id: int, attempt: int = 1, max_attempts: int = 3) -> bool:
    """Download one record; return True on success."""
    for try_num in range(1, max_attempts + 1):
        try:
            wfdb.dl_database(
                db_dir="mitdb",
                dl_dir=str(DEST),
                records=[str(record_id)],
                annotators=["atr"],
            )
            if is_complete(record_id):
                return True
            else:
                print(f"  [{record_id}] Download reported success but files missing — retrying")
                remove_partial(record_id)
        except Exception as exc:
            print(f"  [{record_id}] Attempt {try_num}/{max_attempts} failed: {exc}")
            remove_partial(record_id)
            if try_num < max_attempts:
                time.sleep(3 * try_num)   # back-off before retry
    return False


def main() -> None:
    print("=" * 60)
    print("MIT-BIH Arrhythmia Database — Full Download")
    print(f"Destination: {DEST}")
    print("=" * 60)

    # Classify current state
    already_complete = [r for r in ALL_RECORDS if is_complete(r)]
    needs_download   = [r for r in ALL_RECORDS if not is_complete(r)]

    print(f"\nAlready complete : {len(already_complete)}/48")
    print(f"Need downloading : {len(needs_download)}/48")
    if needs_download:
        print(f"  Records: {', '.join(str(r) for r in needs_download)}\n")

    if not needs_download:
        print("All 48 records already present — nothing to do.")
        _final_audit()
        return

    # Remove stale partial files for records we need to (re)download
    for rid in needs_download:
        remove_partial(rid)

    # Download
    failed = []
    for idx, rid in enumerate(needs_download, 1):
        print(f"[{idx:2d}/{len(needs_download)}] Downloading record {rid} ...", end=" ", flush=True)
        t0 = time.time()
        ok = download_record(rid)
        elapsed = time.time() - t0
        if ok:
            print(f"OK  ({elapsed:.1f}s)")
        else:
            print(f"FAILED after 3 attempts")
            failed.append(rid)

    # Final audit
    print()
    _final_audit()

    if failed:
        print(f"\nWARNING: {len(failed)} record(s) could not be downloaded: {failed}")
        print("Check your internet connection and re-run this script.")
        sys.exit(1)
    else:
        print("\nAll 48 records downloaded successfully.")
        sys.exit(0)


def _final_audit() -> None:
    print("\n--- Final Audit ---")
    complete = [r for r in ALL_RECORDS if is_complete(r)]
    missing  = [r for r in ALL_RECORDS if not is_complete(r)]
    print(f"Complete : {len(complete)}/48  → {', '.join(str(r) for r in complete)}")
    if missing:
        print(f"Missing  : {len(missing)}/48  → {', '.join(str(r) for r in missing)}")


if __name__ == "__main__":
    main()
