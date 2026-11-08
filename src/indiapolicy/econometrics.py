"""Estimation helpers: OLS with HAC errors, linear combinations, delta-method
long-run ratios and standard diagnostics."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_breusch_godfrey
from statsmodels.tsa.stattools import adfuller


def ols(y, X, hac=None, const=True):
    """OLS on the complete-case sample; Newey-West HAC(hac) covariance if hac is set."""
    X = sm.add_constant(X) if const else X
    y = y.reindex(X.index)
    ok = y.notna() & X.notna().all(axis=1)
    m = sm.OLS(y[ok], X[ok])
    return m.fit(cov_type="HAC", cov_kwds={"maxlags": hac}) if hac else m.fit()


def lincomb(res, w):
    """Linear combination w'b with standard error and 95% CI."""
    w = pd.Series(w).reindex(res.params.index).fillna(0.0)
    est = float(w @ res.params)
    se = float(np.sqrt(w @ res.cov_params() @ w))
    return dict(est=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se)


def ratio_sum(res, nums, den):
    """Long-run response sum(b_nums) / (1 - b_den) with delta-method SE."""
    nums = [nums] if isinstance(nums, str) else list(nums)
    b = sum(res.params[n] for n in nums)
    r = res.params[den]
    g = np.array([1 / (1 - r)] * len(nums) + [b / (1 - r) ** 2])
    idx = nums + [den]
    est = b / (1 - r)
    se = float(np.sqrt(g @ res.cov_params().loc[idx, idx].values @ g))
    return dict(est=est, se=se, lo=est - 1.96 * se, hi=est + 1.96 * se)


def bg_pvalue(res, nlags):
    """Breusch-Godfrey LM test p-value (refit by plain OLS on the same design)."""
    return float(acorr_breusch_godfrey(sm.OLS(res.model.endog, res.model.exog).fit(), nlags=nlags)[1])


def adf(series, freq):
    """ADF test with constant. Quarterly: AIC lag selection; annual: one fixed lag."""
    s = series.dropna()
    r = adfuller(s, autolag="AIC") if freq == "Q" else adfuller(s, maxlag=1, autolag=None)
    return dict(stat=float(r[0]), p=float(r[1]), lags=int(r[2]), nobs=int(r[3]))
