"""Run one configured experiment (smoke / pilot / full) -> results/<name>/ with manifest.

usage: python scripts/run_experiment.py configs/smoke.json [--workers N] [--out DIR]
Re-running the same command resumes from completed chunks.
"""
import _bootstrap
import argparse
import pathlib

from ccsim.config import load_config
from ccsim.manifest import write_manifest
from ccsim.runner import run_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("config")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = load_config(a.config)
    out = pathlib.Path(a.out) if a.out else _bootstrap.ROOT / "results" / cfg["name"]
    workers = a.workers or cfg["workers"]
    counts, wall = run_experiment(cfg, out, workers=workers)
    write_manifest(cfg, out, _bootstrap.ROOT, counts, wall, workers)
    print(f"[done] {counts}  wall={wall:.1f}s  manifest -> {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
