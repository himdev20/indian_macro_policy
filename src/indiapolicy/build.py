"""Construct the quarterly analysis panel: base-year splice, seasonal adjustment,
inflation and policy-rate aggregation."""
import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import STL

from .config import BASE_BREAK


def fy_of(ts):
    """Indian fiscal-year label = calendar year in which March falls."""
    return ts.year + 1 if ts.month >= 4 else ts.year


def splice_quarterly(D):
    """Ratio-splice the old-base (pre-2022Q2) segment onto the 2022-23 base.

    2022-23 is published under both bases (RBI Handbook Tables 1-2), which
    identifies a level ratio k = new/old. The old segment is rescaled so that
    FY2021-22 equals the official old-base value times k. By construction the
    annual 2022-23 growth rate then equals the official old-base print.
    """
    Q = D.Q.copy()
    Q["fy"] = [fy_of(t) for t in Q.index]
    old = Q.index < BASE_BREAK
    S22 = Q.loc[Q.fy == 2022]  # FY2021-22 quarters (old-base segment)
    N = D.NAS
    k_r = N["rgdp_new_FY23"] / N["rgdp_old_FY23"]
    f_r = N["rgdp_old_FY22"] * k_r * 10 / S22.rgdp.sum()          # crore -> INR mn
    k_n = N["ngdp_new"][2023] / N["ngdp_old"][2023]
    f_n = N["ngdp_old"][2022] * k_n * 10 / S22.ngdp.sum()
    pf22_old = N["pfce_pc_old_FY22"] * N["pop_FY22_lakh"] * 1e5 / 1e7  # Rs crore
    k_c = N["pfce_pc_new_FY23"] / N["pfce_pc_old_FY23"]
    f_c = pf22_old * k_c * 10 / S22.rpfce.sum()
    Q.loc[old, "rgdp"] *= f_r
    Q.loc[old, "ngdp"] *= f_n
    Q.loc[old, "rpfce"] *= f_c
    Q.attrs["splice_factors"] = dict(real=f_r, nominal=f_n, pfce=f_c, k_real=k_r, k_nom=k_n, k_pfce=k_c)
    return Q


def seasonal_adjust(logy, seg_break=BASE_BREAK):
    """Seasonally adjust a log-level series that has a base-year break.

    Old-base segment (72 quarters): robust STL (period 4, seasonal window 7),
    which allows slowly evolving seasonality and down-weights pandemic outliers.
    New-base segment (~17 quarters): too short for STL, so fixed quarter effects
    are estimated by OLS on a linear trend plus sum-to-zero quarter dummies.
    """
    sa = logy.copy()
    old = logy.index < seg_break
    s_old = logy[old].dropna()
    sa.loc[s_old.index] = s_old - STL(s_old, period=4, seasonal=7, robust=True).fit().seasonal
    s_new = logy[~old].dropna()
    qd = pd.get_dummies(s_new.index.quarter).astype(float).values
    X = np.column_stack([np.arange(len(s_new)), qd])
    b = np.linalg.lstsq(X, s_new.values, rcond=None)[0]
    sf = b[1:] - b[1:].mean()
    sa.loc[s_new.index] = s_new.values - qd @ sf
    return sa


def daily_repo(D, end="2026-12-31"):
    """Policy repo rate as a daily step function."""
    idx = pd.date_range("2013-01-29", end, freq="D")
    s = pd.Series(np.nan, index=idx)
    for d, r in D.repo_changes.items():
        s[d] = r
    return s.ffill()


def quarterly_panel(D):
    """Quarterly panel 2004Q2-2026Q3. Returns (panel, monthly CPI, daily repo)."""
    Q = splice_quarterly(D)
    out = pd.DataFrame(index=Q.index)
    out["fy"] = Q["fy"]
    for v in ["rgdp", "ngdp", "rpfce"]:
        out["l" + v] = np.log(Q[v])
        out["sa_l" + v] = seasonal_adjust(out["l" + v])
    out["g_yoy"] = 100 * (out["sa_lrgdp"] - out["sa_lrgdp"].shift(4))
    out["g_qoq"] = 100 * (out["sa_lrgdp"] - out["sa_lrgdp"].shift(1))
    out["c_qoq"] = 100 * (out["sa_lrpfce"] - out["sa_lrpfce"].shift(1))
    out["c_yoy"] = 100 * (out["sa_lrpfce"] - out["sa_lrpfce"].shift(4))
    out["g_yoy_nsa"] = 100 * (out["lrgdp"] - out["lrgdp"].shift(4))
    cpi = D.cpi_yoy
    out = out.reindex(pd.date_range("2004-04-01", "2026-07-01", freq="QS-JAN"))
    out["pi"] = cpi.resample("QS").mean().reindex(out.index)
    rep = daily_repo(D)
    out["repo_avg"] = rep.resample("QS").mean().reindex(out.index)
    out["repo_end"] = rep.resample("QS").last().reindex(out.index)
    out.attrs = Q.attrs
    return out, cpi, rep
