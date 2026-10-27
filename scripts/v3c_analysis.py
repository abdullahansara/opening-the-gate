"""Analysis of Simulation 3c (results/v3c_anti, results/v3c_store), exploratory; plan in docs/v3c_plan.md.
usage: python scripts/v3c_analysis.py [--nboot 400]
"""
import _bootstrap
import argparse
import json

import numpy as np
import pandas as pd

import analyze_v2 as A

ap = argparse.ArgumentParser(); ap.add_argument("--nboot", type=int, default=400); a = ap.parse_args()
R = _bootstrap.ROOT / "results"; OUT = R / "v3c_anti" / "analysis"; OUT.mkdir(parents=True, exist_ok=True)
S = {}


def load(name):
    r = pd.read_csv(R / name / "runs.csv"); return r[r.stage == "design"].copy()


def block(d, pairs, label):
    pv = A.pivot(d); rows = [A.contrast_row(n, pv, x, y, a.nboot) for n, x, y in pairs]
    out = pd.DataFrame(rows).round(4); out.insert(0, "analysis", label); return out


# ---------------------------------------------------------- composition control
an = load("v3c_anti"); old = load("v2_confirm"); old = old[old.variant == "internal"].set_index("run_id").usefulness
x = an[an.variant == "internal"].set_index("run_id").usefulness; j = x.index.intersection(old.index)
S["N0"] = {"n_compared": int(len(j)), "max_abs_diff": float((x[j] - old[j]).abs().max())}
pairs = [("antigate - random online", "internal_antigate", "internal_random_online"), ("antigate - internal (surprise gate)", "internal_antigate", "internal"), ("internal - random online", "internal", "internal_random_online")]
C = block(an, pairs, "composition: all environments")
for e in sorted(an.env.unique()):
    C = pd.concat([C, block(an[an.env == e], pairs, f"composition: {e}")], ignore_index=True)
C.to_csv(OUT / "C1_composition_contrasts.csv", index=False)
M = an.groupby("variant").agg(usefulness=("usefulness", "mean"), admitted=("n_admitted", "mean"), surprising_admitted=("n_admitted_surprising", "mean"), surprising=("n_surprising", "mean"), outliers_admitted=("n_admitted_outliers", "mean"), false_terms=("n_false_terms", "mean"), recovered=("exact_recovery", "mean")).round(3)
M["share_admitted_surprising"] = (M.surprising_admitted / M.admitted).round(3)
Me = an.groupby(["env", "variant"]).agg(usefulness=("usefulness", "mean"), admitted=("n_admitted", "mean"), surprising_admitted=("n_admitted_surprising", "mean")).round(3)
M.to_csv(OUT / "C1_composition_means.csv"); Me.to_csv(OUT / "C1_composition_means_by_env.csv")
S["composition_contrasts"] = C.to_dict(orient="records"); S["composition_means"] = M.to_dict(orient="index")

# ---------------------------------------------------------- storage test
st = load("v3c_store")
pairs = [("storage: internal_big - internal", "internal_big", "internal"), ("storage with guidance: internal_big_guided - internal_guided", "internal_big_guided", "internal_guided"),
         ("guidance: internal_guided - internal", "internal_guided", "internal"), ("guidance with big store: internal_big_guided - internal_big", "internal_big_guided", "internal_big"),
         ("workspace vs guided internal: interactive - internal_guided", "interactive", "internal_guided"), ("workspace vs guided big store: interactive - internal_big_guided", "interactive", "internal_big_guided"),
         ("passive - internal_big", "passive", "internal_big")]
Cs = block(st, pairs, "storage: all environments")
for e in sorted(st.env.unique()):
    Cs = pd.concat([Cs, block(st[st.env == e], pairs, f"storage: {e}")], ignore_index=True)
Cs.to_csv(OUT / "C2_storage_contrasts.csv", index=False)
Ms = st.groupby("variant").agg(usefulness=("usefulness", "mean"), median=("usefulness", "median"), fail=("usefulness", lambda s: (s < .5).mean()), admitted=("n_admitted", "mean"), recovered=("exact_recovery", "mean")).round(3)
Ms.to_csv(OUT / "C2_storage_means.csv")
Mw = st.assign(wband=pd.cut(st.window, [0, 36, 60, 100], labels=["24-36", "48-60", "72-96"])).groupby(["wband", "variant"], observed=True).usefulness.mean().unstack().round(3)
Mw.to_csv(OUT / "C2_storage_by_window.csv")
S["storage_contrasts"] = Cs.to_dict(orient="records"); S["storage_means"] = Ms.to_dict(orient="index")
(OUT / "v3c_stats.json").write_text(json.dumps(S, indent=1, default=float)); print("done ->", OUT)
print(M.to_string()); print(C[C.analysis == "composition: all environments"].to_string()); print(C.iloc[3:].to_string())
print(Ms.to_string()); print(Mw.to_string()); print(Cs[Cs.analysis == "storage: all environments"].to_string())
