"""Rebuild every table, statistic and figure from the raw logs of one run.

usage: python scripts/analyze.py results/<run>      (reads results/<run>/manifest.json for the config)
"""
import _bootstrap
import json
import pathlib
import sys

from ccsim.analysis import run_analysis
from ccsim.manifest import sha256_file


def main():
    out = pathlib.Path(sys.argv[1])
    man = json.loads((out / "manifest.json").read_text(encoding="utf-8"))
    res = run_analysis(out, man["config"], name=man["config_name"])
    an = out / "analysis"
    sums = {str(p.relative_to(an)): sha256_file(p) for p in sorted(an.rglob("*")) if p.is_file()
            and p.name != "analysis_checksums.json" and p.suffix in (".csv", ".json")}
    (an / "analysis_checksums.json").write_text(json.dumps(sums, indent=2), encoding="utf-8")
    print(f"[analysis] done -> {an}")
    for h in ("H1", "H2", "H3", "H4", "H5"):
        print(f"  {h}: supported={res[h].get('supported')}")


if __name__ == "__main__":
    main()
