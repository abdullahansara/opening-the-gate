"""Crossed-bootstrap intervals for the regression-based primary tests (C3a, C3b, C5) so that both interval types can be reported.
usage: python scripts/v2x_boot_regression.py [--nboot 200]  -> results/v2_confirm/analysis_v2x/R12_bootstrap_regression.csv"""
import _bootstrap
import argparse

import numpy as np
import pandas as pd
import patsy
import statsmodels.api as sm
import statsmodels.formula.api as smf

import analyze_v2 as A

ap = argparse.ArgumentParser(); ap.add_argument("--nboot", type=int, default=200); a = ap.parse_args()
RUN = _bootstrap.ROOT / "results" / "v2_confirm"; runs, d = A.load(RUN)
s = d[d.variant == "internal"].copy(); s["gc"] = s.g - .5
y, X = patsy.dmatrices("n_false_terms ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)", s, return_type="dataframe"); cols = list(X.columns)
fit = lambda w: sm.GLM(y.values.ravel(), X.values, family=sm.families.Poisson(), freq_weights=w).fit().params
base = fit(np.ones(len(s))); rows = []
bs = np.array([fit(A.crossed_weights(s)) for _ in range(a.nboot)])
mm = smf.glm("n_false_terms ~ gc*lam_c + gc*margin_c + f + window + m_pool + C(env)", s, family=sm.families.Poisson()).fit(cov_type="cluster", cov_kwds={"groups": A.g2(s)}); ci = mm.conf_int()
for t in ("lam_c", "margin_c"):
    k = cols.index(t); lo, hi = np.percentile(bs[:, k], [2.5, 97.5])
    rows.append({"test": "C3 " + t + " (Poisson, at g = .5)", "estimate": float(base[k]), "lo_twoway": float(ci.loc[t, 0]), "hi_twoway": float(ci.loc[t, 1]), "lo_boot": float(lo), "hi_boot": float(hi), "lo": float(min(ci.loc[t, 0], lo)), "hi": float(max(ci.loc[t, 1], hi))})
# C5 slope
s5 = d.copy(); s5["fp_nonproxy"], s5["fp_proxy"] = A.proxy_counts(s5); s5["n_valid"] = s5.n_sel_terms - s5.n_false_terms
y5, X5 = patsy.dmatrices("usefulness ~ fp_nonproxy + fp_proxy + n_valid + n_true_terms + C(env)", s5, return_type="dataframe"); c5 = list(X5.columns)
m5 = A.ols2("usefulness ~ fp_nonproxy + fp_proxy + n_valid + n_true_terms + C(env)", s5); ci5 = m5.conf_int()
b5 = np.array([A.wls(X5.values, y5.values.ravel(), A.crossed_weights(s5)) for _ in range(a.nboot)])
for t in ("fp_nonproxy", "fp_proxy", "n_valid"):
    k = c5.index(t); lo, hi = np.percentile(b5[:, k], [2.5, 97.5])
    rows.append({"test": "C5 " + t, "estimate": float(m5.params[t]), "lo_twoway": float(ci5.loc[t, 0]), "hi_twoway": float(ci5.loc[t, 1]), "lo_boot": float(lo), "hi_boot": float(hi), "lo": float(min(ci5.loc[t, 0], lo)), "hi": float(max(ci5.loc[t, 1], hi))})
out = pd.DataFrame(rows).round(4); out.to_csv(RUN / "analysis_v2x" / "R12_bootstrap_regression.csv", index=False); print(out.to_string(index=False))
