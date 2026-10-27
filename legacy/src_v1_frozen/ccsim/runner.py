"""Experiment runner: executes tasks in parallel, checkpoints per chunk, never drops failed runs."""
import json
import multiprocessing as mp
import os
import pathlib
import time
import traceback
import warnings

import numpy as np
import pandas as pd

from .agents import Agent
from .config import config_hash
from .design import build_tasks
from .grammar import get_library
from .metrics import heldout_eval, oracle_ll, structure_metrics, proposal_metrics, recovery_metrics
from .rng import stream, ENV_IDS, ARTIFACT_IDS
from .world import make_world

_WORLD_CACHE = {}


def _lower_priority():
    """Keep the machine responsive: run workers at below-normal priority (Windows) / nice (POSIX)."""
    try:
        if os.name == "nt":
            import ctypes
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)
        else:
            os.nice(5)
    except Exception:
        pass


def _get_world(cfg, env, wid):
    key = (config_hash(cfg["world"]) + str(cfg["master_seed"]), env, wid)
    if key not in _WORLD_CACHE:
        if len(_WORLD_CACHE) > 64:
            _WORLD_CACHE.clear()
        w = make_world(cfg, env, wid)
        _WORLD_CACHE[key] = (w, get_library(cfg["world"]["d"]).features(w.X_test))
    return _WORLD_CACHE[key]


def run_agent(cfg, world, Phi_test, stage, label, spec, key_id, adm_key, design_id):
    lib = get_library(cfg["world"]["d"])
    T = cfg["world"]["T"]
    master, env_id, wid = cfg["master_seed"], ENV_IDS[world.env], world.world_id
    art_id = ARTIFACT_IDS[spec["artifact"]]
    u = stream(master, "admission", env_id, wid, adm_key).random(T)
    rp = stream(master, "proposal", env_id, wid, key_id, art_id)
    rr = stream(master, "retrieval", env_id, wid, key_id, art_id)
    agent = Agent(spec, cfg["agent"], lib, T, u, rp, rr)
    run_id = f"{stage}|{label}|{design_id}|{world.env}|{wid}" + (f"|{spec['artifact']}" if stage == "design" else "")
    status, err_msg = "ok", ""
    curves = []
    ck = cfg["run"]["ckpt_every"]
    t0 = time.perf_counter()
    t_done = T
    with warnings.catch_warnings(record=True) as wlist:
        warnings.simplefilter("always")
        try:
            for t in range(T):
                x, y = world.stream.get(t)
                agent.observe(t, x, y)
                if (t + 1) % ck == 0:
                    ev = heldout_eval(agent, Phi_test, world)
                    lo = max(0, t + 1 - ck)
                    curves.append({"run_id": run_id, "t": t + 1, **ev,
                                   "preq_mse": float(agent.preq_err[lo:t + 1].mean())})
                if time.perf_counter() - t0 > cfg["run"]["timeout_s"]:
                    status, t_done = "timeout", t + 1
                    break
        except Exception as e:  # kept in the data with a status code, never discarded
            status, err_msg = "error", f"{type(e).__name__}: {e}"[:200]
            t_done = int(agent.admitted.size)
            traceback.print_exc()
    elapsed = time.perf_counter() - t0
    if status == "ok" and wlist:
        status = "warn"
        err_msg = str(wlist[0].message)[:200]
    row = {"run_id": run_id, "stage": stage, "variant": label, "design_id": design_id, "env": world.env,
           "world_id": wid, "artifact": spec["artifact"], "mode": spec.get("mode", "local"),
           "g": spec["g"], "f": spec["f"], "c": spec["c"], "pi": spec["pi"], "m": spec["m"],
           "free_ops": int(bool(spec.get("free_ops", False))), "status": status, "error": err_msg,
           "trials_done": t_done, "elapsed_s": elapsed, "program_hash": world.program_hash,
           "noise_sd": world.noise_sd}
    try:
        ev = heldout_eval(agent, Phi_test, world)
        ev["oracle_ll"] = oracle_ll(world)
        ev["ll_gap"] = ev["oracle_ll"] - ev["heldout_ll"]
        ev["usefulness"] = float(np.clip(ev["r2_true"], -1.0, 1.0))
        row.update(ev)
        row.update(structure_metrics(lib, agent.map, world))
        row.update(proposal_metrics(lib, agent, world))
        row.update(recovery_metrics(agent, world, cfg["run"]["recover_window"], cfg["run"]["recover_mult"]))
    except Exception as e:
        row["status"] = "error"; row["error"] = f"metrics: {type(e).__name__}: {e}"[:200]
    c = agent.cnt
    row.update({
        "n_switches": agent.n_switch, "n_revisions": c["revisions"],
        "n_surprising": int(agent.surprising.sum()),
        "n_admitted": int(agent.admitted.sum()),
        "n_admitted_surprising": int((agent.admitted & agent.surprising).sum()),
        "n_admitted_outliers": int((agent.admitted & world.outlier_mask).sum()),
        "n_outliers": int(world.outlier_mask.sum()),
        "evals_prop": c["evals_prop"], "evals_pool": c["evals_pool"], "writes": c["writes"],
        "reads": c["reads"], "annotate_ops": c["annotate"], "compare_ops": c["compare"],
        "combine_ops": c["combine"], "artifact_cost": c["artifact_cost"], "prop_cost": c["spent_cost"],
        "cost_units": c["artifact_cost"] + c["spent_cost"],
        "preq_mse": float(agent.preq_err[:t_done].mean()) if t_done else np.nan,
    })
    return row, curves


def execute_chunk(args):
    cfg, chunk_id, tasks, outdir = args
    _lower_priority()
    outdir = pathlib.Path(outdir)
    rows, curves, worlds, seen = [], [], [], set()
    for task in tasks:
        world, Phi_test = _get_world(cfg, task["env"], task["world_id"])
        if (task["env"], task["world_id"]) not in seen:
            seen.add((task["env"], task["world_id"]))
            worlds.append({"env": task["env"], "world_id": task["world_id"], "program_hash": world.program_hash,
                           "spec_json": json.dumps(world.spec, sort_keys=True)})
        for label, spec, key_id, adm_key in task["agents"]:
            design_id = key_id if task["stage"] == "design" else (key_id if task["stage"] == "ablation" else key_id)
            r, cv = run_agent(cfg, world, Phi_test, task["stage"], label, spec, key_id, adm_key, design_id)
            rows.append(r); curves += cv
    tmp = outdir / "chunks"
    tmp.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(tmp / f"runs_{chunk_id:05d}.csv", index=False)
    pd.DataFrame(curves).to_csv(tmp / f"curves_{chunk_id:05d}.csv", index=False)
    pd.DataFrame(worlds).to_csv(tmp / f"worlds_{chunk_id:05d}.csv", index=False)
    (tmp / f"done_{chunk_id:05d}.flag").write_text(str(len(rows)))
    return chunk_id, len(rows)


def run_experiment(cfg, outdir, workers=None, stages=("design", "ablation", "baseline"), log=print):
    outdir = pathlib.Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(cfg, stages)
    n = cfg["chunk_tasks"]
    chunks = [tasks[i:i + n] for i in range(0, len(tasks), n)]
    todo = [(cfg, i, ch, str(outdir)) for i, ch in enumerate(chunks)
            if not (outdir / "chunks" / f"done_{i:05d}.flag").exists()]
    workers = workers or cfg["workers"]
    n_runs = sum(len(t["agents"]) for t in tasks)
    log(f"[run] {n_runs} runs in {len(tasks)} tasks / {len(chunks)} chunks; {len(chunks) - len(todo)} already done; workers={workers}")
    t0 = time.perf_counter()
    if todo:
        if workers <= 1:
            for a in todo:
                execute_chunk(a)
        else:
            ctx = mp.get_context("spawn")
            with ctx.Pool(workers, initializer=_lower_priority) as pool:
                for k, (cid, nr) in enumerate(pool.imap_unordered(execute_chunk, todo), 1):
                    if k % max(1, len(todo) // 10) == 0 or k == len(todo):
                        log(f"[run] chunk {k}/{len(todo)} done ({time.perf_counter() - t0:.0f}s)")
    wall = time.perf_counter() - t0
    return merge_chunks(outdir, len(chunks)), wall


def merge_chunks(outdir, n_chunks):
    outdir = pathlib.Path(outdir)
    parts = {k: [] for k in ("runs", "curves", "worlds")}
    for i in range(n_chunks):
        for k in parts:
            f = outdir / "chunks" / f"{k}_{i:05d}.csv"
            try:
                parts[k].append(pd.read_csv(f))
            except pd.errors.EmptyDataError:
                pass
    out = {}
    for k, lst in parts.items():
        df = pd.concat(lst, ignore_index=True)
        key = "run_id" if k != "worlds" else ["env", "world_id"]
        df = df.sort_values(key if k != "curves" else ["run_id", "t"]).drop_duplicates(
            subset=(["run_id", "t"] if k == "curves" else key)).reset_index(drop=True)
        df.to_csv(outdir / f"{k}.csv", index=False)
        out[k] = len(df)
    return out
