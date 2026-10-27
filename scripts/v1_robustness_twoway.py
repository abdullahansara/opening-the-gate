"""Study 1 constants sweep re-analysed with two-way (worlds x design points) cluster-robust inference.
usage: python scripts/v1_robustness_twoway.py   -> results/robustness/robustness_twoway.csv
"""
import _bootstrap
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

warnings.filterwarnings("ignore")
ROOT = _bootstrap.ROOT
COV = "C(env) + f + c + pi + m"
rows = []
for d_ in sorted((ROOT / "results" / "robustness").glob("robust_*")):
    f = d_ / "runs.csv"
    if not f.exists():
        continue
    r = pd.read_csv(f); d = r[r.stage == "design"].copy()
    d["wk"] = pd.factorize(d.env + "_" + d.world_id.astype(str))[0]; d["dk"] = d.design_id
    d["weak"] = (d.c <= 1 / 3).astype(int); d["strong"] = (d.c >= 2 / 3).astype(int)
    g2 = lambda x: np.column_stack([x.wk.values, x.dk.values])
    fit = lambda form, x: smf.ols(form, x).fit(cov_type="cluster", cov_kwds={"groups": g2(x)})

    def conf(form, x, term):
        """Two-way cluster-robust 95% CI; if the two-way variance is not positive (few clusters), use the larger of the
        world-only and design-point-only standard errors (conservative)."""
        m2 = fit(form, x); est = float(m2.params[term]); se = float(m2.bse[term])
        if not np.isfinite(se):
            ses = [float(smf.ols(form, x).fit(cov_type="cluster", cov_kwds={"groups": x[c].values}).bse[term]) for c in ("wk", "dk")]
            se = max(ses)
        return est, est - 1.96 * se, est + 1.96 * se, m2
    adeq = d[(d.c >= .5) & (d.f >= .3)]; d2 = d[(d.weak == 1) | (d.strong == 1)]
    out = {"setting": d_.name.replace("robust_", "")}
    e, lo, hi, _ = conf(f"usefulness ~ g + I(g**2) + C(artifact) + {COV}", adeq, "I(g ** 2)")
    out.update(curv=e, curv_lo=lo, curv_hi=hi)
    e, lo, hi, _ = conf(f"false_discovery ~ g*weak + C(artifact) + {COV}", d2, "g:weak")
    out.update(h2=e, h2_lo=lo, h2_hi=hi)
    m = fit(f"usefulness ~ g * C(artifact, Treatment('internal')) + {COV}", d)
    inter = [n for n in m.params.index if n.startswith("g:C(artifact")]; R = np.zeros((3, len(m.params)))
    for i, n in enumerate(inter):
        R[i, list(m.params.index).index(n)] = 1
    out["h4_p"] = float(m.wald_test(R, scalar=True).pvalue)
    pv = d.pivot_table(index=["design_id", "env", "world_id"], columns="artifact", values="usefulness").dropna().reset_index()
    pv["wk"] = pd.factorize(pv.env + "_" + pv.world_id.astype(str))[0]; pv["dk"] = pv.design_id
    for k in ("passive", "internal_big", "internal"):
        pv["dd"] = pv["interactive"] - pv[k]; e, lo, hi, _ = conf("dd ~ 1", pv, "Intercept")
        out.update({f"h3_{k}": e, f"h3_{k}_lo": lo, f"h3_{k}_hi": hi})
    rows.append(out)
t = pd.DataFrame(rows); t.to_csv(ROOT / "results" / "robustness" / "robustness_twoway.csv", index=False)
n = len(t)
print(f"{n} settings")
print("curvature negative:", int((t.curv < 0).sum()), "| CI excludes 0 (negative):", int((t.curv_hi < 0).sum()))
print("H2 coef > 0:", int((t.h2 > 0).sum()), "| CI above 0:", int((t.h2_lo > 0).sum()))
print("H3 CI above 0 passive/matched/internal:", int((t.h3_passive_lo > 0).sum()), int((t.h3_internal_big_lo > 0).sum()), int((t.h3_internal_lo > 0).sum()))
print("H4 p<.05:", int((t.h4_p < .05).sum()))
