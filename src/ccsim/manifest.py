"""Run manifest: provenance and output checksums (spec 10)."""
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import sys
from importlib import metadata

from .config import config_hash

PKGS = ["numpy", "scipy", "pandas", "matplotlib", "statsmodels", "patsy", "pytest"]


def sha256_file(p) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for blk in iter(lambda: f.read(1 << 20), b""):
            h.update(blk)
    return h.hexdigest()


def source_hash(src_dir) -> str:
    h = hashlib.sha256()
    for p in sorted(pathlib.Path(src_dir).rglob("*.py")):
        h.update(p.name.encode()); h.update(p.read_bytes().replace(b"\r\n", b"\n"))
    return h.hexdigest()


def _git_hash(root):
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:
        return None


def _cpu():
    try:
        if platform.system() == "Windows":
            out = subprocess.run(["powershell", "-NoProfile", "-Command",
                                  "(Get-CimInstance Win32_Processor).Name"], capture_output=True, text=True, timeout=20)
            return out.stdout.strip()
    except Exception:
        pass
    return platform.processor()


def write_manifest(cfg, outdir, root, counts, wall_s, workers, extra=None):
    import os
    outdir = pathlib.Path(outdir)
    versions = {}
    for p in PKGS:
        try:
            versions[p] = metadata.version(p)
        except Exception:
            versions[p] = None
    files = {}
    for name in ("runs.csv", "curves.csv", "worlds.csv"):
        f = outdir / name
        if f.exists():
            files[name] = sha256_file(f)
    chunk_sums = {p.name: sha256_file(p) for p in sorted((outdir / "chunks").glob("*.csv"))}
    from .config import world_offset
    man = {
        "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "config_name": cfg["name"], "config_hash": config_hash(cfg), "config": cfg,
        "source_sha256": source_hash(pathlib.Path(root) / "src"),
        "git_commit": _git_hash(root),
        "seeds": {"master_seed": cfg["master_seed"], "seed_split": cfg["seed_split"],
                  "world_id_offset": world_offset(cfg), "design_worlds": cfg["design"]["n_worlds"],
                  "ablation_worlds": cfg["ablation"]["n_worlds"], "baseline_worlds": cfg["baseline_worlds"]},
        "python": sys.version, "platform": platform.platform(), "cpu": _cpu(),
        "logical_cores": os.cpu_count(), "gpu": None, "workers": workers,
        "package_versions": versions, "counts": counts, "wall_seconds_this_invocation": round(wall_s, 2),
        "output_sha256": files, "chunk_sha256": chunk_sums,
    }
    if extra:
        man.update(extra)
    (outdir / "manifest.json").write_text(json.dumps(man, indent=2, sort_keys=True), encoding="utf-8")
    return man


def verify_manifest(outdir) -> list:
    """Return list of (file, ok) after recomputing checksums."""
    outdir = pathlib.Path(outdir)
    man = json.loads((outdir / "manifest.json").read_text(encoding="utf-8"))
    res = []
    for name, h in man["output_sha256"].items():
        res.append((name, sha256_file(outdir / name) == h))
    for name, h in man["chunk_sha256"].items():
        res.append((f"chunks/{name}", sha256_file(outdir / "chunks" / name) == h))
    return res
