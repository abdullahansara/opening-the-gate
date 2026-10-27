"""Verify a results directory: (1) raw-output checksums vs manifest, (2) analysis rebuilt from raw logs
alone reproduces the stored tables/statistics byte-for-byte.

usage: python scripts/verify.py results/<run> [--skip-rebuild]
"""
import _bootstrap
import json
import pathlib
import shutil
import sys
import tempfile

from ccsim.analysis import run_analysis
from ccsim.manifest import sha256_file, verify_manifest


def main():
    out = pathlib.Path(sys.argv[1]); skip = "--skip-rebuild" in sys.argv
    ok_all = True
    res = verify_manifest(out)
    bad = [n for n, ok in res if not ok]
    print(f"[verify] raw outputs: {len(res) - len(bad)}/{len(res)} checksums match manifest")
    ok_all &= not bad
    if not skip:
        man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
        stored = json.loads((out / "analysis" / "analysis_checksums.json").read_text())
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            for f in ("runs.csv", "curves.csv", "worlds.csv", "manifest.json"):
                shutil.copy(out / f, td / f)
            run_analysis(td, man["config"], name=man["config_name"])
            diff = []
            for rel, h in stored.items():
                p = td / "analysis" / rel
                if not p.exists() or sha256_file(p) != h:
                    diff.append(rel)
        print(f"[verify] analysis rebuild: {len(stored) - len(diff)}/{len(stored)} files identical to stored")
        for d in diff:
            print("   MISMATCH:", d)
        ok_all &= not diff
    print("[verify] PASS" if ok_all else "[verify] FAIL")
    import datetime
    (out / "verification.json").write_text(json.dumps({
        "passed": bool(ok_all), "analysis_rebuilt": not skip,
        "checked_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "raw_files_checked": len(res), "raw_files_mismatched": bad}, indent=2))
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
