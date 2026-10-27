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
    mu_map = agent.predict_map(Phi_test)
    ll = float(np.mean(gauss_ll(world.y_test, mu, var)))
    mse = float(np.mean((mu - world.y_test) ** 2))
    f_var = float(np.var(world.f_test))
    r2_true = 1.0 - float(np.mean((mu - world.f_test) ** 2)) / f_var      # share of true signal captured
    cover = float(np.mean(np.abs(world.y_test - mu) <= 1.645 * np.sqrt(np.maximum(var, 1e-8))))
    r2_map = 1.0 - float(np.mean((mu_map - world.f_test) ** 2)) / f_var   # prediction from the MAP structure only
    return {"heldout_ll": ll, "heldout_mse": mse, "r2_true": r2_true, "cover90": cover, "r2_map": r2_map}


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


def tailed_metrics(lib, agent, world, burn_in):
    """Simulation 4 measures (tailed worlds only). Error on the informative tail (held-out signal-tail inputs and a fixed probe set at the
    tail's reach), error on the bulk, how much of the signal tail the learner took in, and how well its surprise picks out signal trials."""
    from scipy.stats import rankdata
    mu_p, _ = agent.predict(lib.features(world.X_probe))
    r2_probe = 1.0 - float(np.mean((mu_p - world.f_probe) ** 2)) / float(np.var(world.f_probe))
    mu, _ = agent.predict(lib.features(world.X_test))
    f_var = float(np.var(world.f_test)); st = world.sig_test; err = (mu - world.f_test) ** 2
    tail_nmse = float(err[st].mean() / f_var) if st.sum() >= 10 else np.nan
    bulk_nmse = float(err[~st].mean() / f_var)
    sig = world.sig_mask; adm = agent.admitted
    z = agent.zabs[burn_in:]; sg = sig[burn_in:]; ok = ~world.outlier_mask[burn_in:]       # signal vs ordinary trials (noise trials left out)
    auc = np.nan
    if sg.sum() > 0 and (ok & ~sg).sum() > 0:
        zz = np.concatenate([z[sg], z[ok & ~sg]]); rk = rankdata(zz); n1, n0 = int(sg.sum()), int((ok & ~sg).sum())
        auc = float((rk[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    return {"r2_probe": r2_probe, "tail_nmse": tail_nmse, "bulk_nmse": bulk_nmse, "n_signal": int(sig.sum()),
            "n_signal_admitted": int((adm & sig).sum()), "auc_surprise_signal": auc}


# ---------------------------------------------------------------------------------------- v2 diagnostics
_PROXY = {}


def _proxy_r2(lib, false_id, true_ids):
    """Multiple R^2 of a library term on the true non-constant terms (+ intercept) over N(0, I) inputs (cached, fixed draws)."""
    if "F" not in _PROXY:
        _PROXY["F"] = lib.features(np.random.default_rng(3).standard_normal((20000, lib.d)))
    key = (false_id, tuple(sorted(true_ids)))
    if key not in _PROXY:
        F = _PROXY["F"]
        A = np.column_stack([np.ones(len(F))] + [F[:, k] for k in key[1]])
        y = F[:, false_id]
        coef = np.linalg.lstsq(A, y, rcond=None)[0]
        res = y - A @ coef
        v = y.var()
        _PROXY[key] = float(1.0 - (res @ res / len(y)) / v) if v > 0 else 1.0
    return _PROXY[key]


def v2_diagnostics(lib, agent, world):
    true_nc = [k for k in world.true_struct if k != 0]
    sel_nc = [k for k in agent.map if k != 0]
    false = [k for k in sel_nc if k not in true_nc]
    min_r2 = min((_proxy_r2(lib, k, true_nc) for k in false), default=np.nan)
    # coefficient recovery on the true terms (MAP structure; a missed term counts as coefficient 0)
    coefs = dict(zip(world.spec["final_terms"], world.spec["final_coefs"]))
    state_coef = dict(zip(agent.map, agent.state[agent.map][0]))
    err = []; tru = []
    for k in true_nc:
        name = lib.names[k]
        err.append(state_coef.get(k, 0.0) - coefs[name]); tru.append(coefs[name])
    nrmse = float(np.sqrt(np.mean(np.square(err))) / np.sqrt(np.mean(np.square(tru)))) if tru else np.nan
    w = agent.weights
    out = {"min_false_r2": float(min_r2), "coef_nrmse": nrmse, "max_weight": float(w.max()), "eff_n_hyp": float(1.0 / np.sum(w ** 2)),
           "n_pool": int(len(w)), "n_switch_rev": agent.n_switch_rev, "n_term_changes": agent.n_term_changes}
    for tt in (100, 160):
        shares = [np.mean(act < world.shift_t) for (t_, act) in agent.active_hist if t_ <= tt - 1] if world.shift_t is not None else []
        out[f"pre_shift_share_t{tt}"] = float(shares[-1]) if shares else np.nan
    return out
