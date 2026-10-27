"""Dependent measures, computed from raw agent behaviour and held-out data (spec 7).

Primary usefulness is evaluated on unseen held-out data and never uses prediction error, novelty,
switching or artifact use as a *definition* of creativity.
"""
import math
import numpy as np

LOG2PI = math.log(2 * math.pi)


def gauss_ll(y, mu, var):
    var = np.maximum(var, 1e-8)
    return -0.5 * (LOG2PI + np.log(var) + (y - mu) ** 2 / var)


def heldout_eval(agent, Phi_test, world):
    mu, var = agent.predict(Phi_test)
    ll = float(np.mean(gauss_ll(world.y_test, mu, var)))
    mse = float(np.mean((mu - world.y_test) ** 2))
    f_var = float(np.var(world.f_test))
    r2_true = 1.0 - float(np.mean((mu - world.f_test) ** 2)) / f_var      # share of true signal captured
    cover = float(np.mean(np.abs(world.y_test - mu) <= 1.645 * np.sqrt(np.maximum(var, 1e-8))))
    return {"heldout_ll": ll, "heldout_mse": mse, "r2_true": r2_true, "cover90": cover}


def oracle_ll(world):
    return float(np.mean(gauss_ll(world.y_test, world.f_test, world.noise_sd ** 2)))


def structure_metrics(lib, map_struct, world):
    true_nc = {k for k in world.true_struct if k != 0}
    sel_nc = {k for k in map_struct if k != 0}
    true_nl = {k for k in true_nc if k in set(lib.nonlinear_ids)}
    found = sel_nc & true_nc
    false = sel_nc - true_nc
    start_nc = set()  # start hypothesis is the constant-only structure
    surpr = [-math.log2(lib.kind_freq[lib.kinds[k]]) for k in sel_nc]
    return {
        "map_struct": lib.struct_names(map_struct),
        "true_struct": lib.struct_names(world.true_struct),
        "n_sel_terms": len(sel_nc),
        "n_true_terms": len(true_nc),
        "true_found_frac": len(found) / max(len(true_nc), 1),
        "nl_found": int(len(true_nl & sel_nc) > 0),
        "nl_found_frac": len(true_nl & sel_nc) / max(len(true_nl), 1),
        "exact_recovery": int(sel_nc == true_nc),
        "n_false_terms": len(false),
        "false_discovery": int(len(false) > 0),
        "fdr": len(false) / max(len(sel_nc), 1) if sel_nc else 0.0,
        "orig_dist": len(sel_nc ^ start_nc) / lib.K,                      # structural distance from start
        "orig_surprisal": float(np.mean(surpr)) if surpr else 0.0,         # rarity under reference proposals
    }


def proposal_metrics(lib, agent, world):
    true_nc = {k for k in world.true_struct if k != 0}
    acc_false = acc_true = rej_true = rej_total = acc_total = 0
    for (_t, added, _dropped, _passed, chosen) in agent.proposal_log:
        add_nc = {k for k in added if k != 0}
        if chosen:
            acc_total += 1
            acc_false += int(len(add_nc - true_nc) > 0)
            acc_true += int(len(add_nc & true_nc) > 0)
        else:
            rej_total += 1
            rej_true += int(len(add_nc & true_nc) > 0)
    return {"n_proposals": len(agent.proposal_log), "n_accepted": acc_total, "n_rejected": rej_total,
            "accepted_false": acc_false, "accepted_true": acc_true, "rejected_true": rej_true}


def recovery_metrics(agent, world, window, mult):
    """Trials to recover after the genuine rule change (shifting env only)."""
    if world.shift_t is None:
        return {"recover_trials": np.nan, "recovered": np.nan}
    err = agent.preq_err
    thr = mult * world.noise_sd ** 2
    T = len(err)
    for t in range(world.shift_t + window - 1, T):
        if err[t - window + 1:t + 1].mean() <= thr:
            return {"recover_trials": float(t - world.shift_t), "recovered": 1.0}
    return {"recover_trials": float(T - world.shift_t), "recovered": 0.0}   # right-censored
