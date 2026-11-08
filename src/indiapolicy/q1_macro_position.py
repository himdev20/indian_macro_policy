"""Q1. India's macroeconomic position: indicators, output gap, inflation equation."""
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from statsmodels.tsa.filters.hp_filter import hpfilter

from .config import PHASES, COVID_START, COVID_END, HAC_Q, INK, C1, C2, C3, C4
from .econometrics import ols, lincomb, bg_pvalue, adf
from .plotting import shade_episodes, save
from .tables import frame_table, reg_table


# ----------------------------------------------------------------- (a) indicators
def indicators(ctx):
    P, cpi_m, repo_d, R = ctx.P, ctx.cpi_m, ctx.repo_d, ctx.R
    s = P.loc["2014-01-01":"2026-07-01"]
    fig, ax = plt.subplots(3, 1, figsize=(7.2, 7.4), sharex=True)
    ax[0].plot(s.index, s.g_yoy, color=C1); ax[0].axhline(0, color=INK, lw=.6)
    ax[0].set_ylabel("Real GDP growth\n(% y/y, quarterly)")
    ax[0].axvspan(pd.Timestamp("2025-04-01"), pd.Timestamp("2026-06-30"), facecolor="none",
                  hatch="///", edgecolor="#9a9a98", lw=0)
    ax[0].text(pd.Timestamp("2025-01-15"), -22, "provisional /\nearly est.", fontsize=6.5, ha="right")
    c = cpi_m.loc["2014":]
    ax[1].plot(c.index, c, color=C2, lw=1.4)
    ax[1].axhspan(2, 6, color="#9a9a98", alpha=.12, lw=0); ax[1].axhline(4, color="#9a9a98", ls="--", lw=.8)
    ax[1].axvline(pd.Timestamp("2026-01-01"), color=INK, ls=":", lw=.8)
    ax[1].text(pd.Timestamp("2026-01-20"), 7.5, "CPI base\n2024=100", fontsize=7)
    ax[1].set_ylabel("CPI-C inflation\n(% y/y, monthly)")
    rp = repo_d.loc["2014-01-01":"2026-10-31"]
    ax[2].step(rp.index, rp.values, where="post", color=C3); ax[2].set_ylabel("Policy repo\nrate (%)")
    for a in ax: shade_episodes(a)
    save(fig, "fig1_indicators")

    tab = pd.DataFrame({k: [P.loc[a:b, "g_yoy"].mean(), P.loc[a:b, "pi"].mean(), P.loc[a:b, "repo_avg"].mean()]
                        for k, (a, b) in PHASES.items()}, index=["Growth", "CPI infl.", "Repo (avg)"]).T
    tab.index.name = "Phase (mean, %)"
    frame_table(tab, "t01_phases", "Table 1. Phase averages (%). Growth: SA real GDP, y/y; repo: quarter average.")
    R["phases"] = tab.round(2).to_dict()
    R["turning_points"] = dict(
        covid=dict(g_2019q4=float(P.loc["2019-10-01", "g_yoy"]), g_trough=float(P.loc["2020-04-01", "g_yoy"]),
                   g_rebound=float(P.loc["2021-04-01", "g_yoy"]), repo_pre=5.15, repo_post=4.00,
                   pi_2020q2=float(P.loc["2020-04-01", "pi"])),
        tight=dict(pi_peak=float(cpi_m.loc["2022"].max()), pi_peak_month=str(cpi_m.loc["2022"].idxmax().date())[:7],
                   repo_pre=4.00, repo_post=6.50),
        ease=dict(pi_trough=float(cpi_m.loc["2025"].min()), pi_trough_month=str(cpi_m.loc["2025"].idxmin().date())[:7],
                  repo_pre=6.50, repo_post=5.25, pi_2025=float(cpi_m.loc["2025"].mean()),
                  pi_latest=float(cpi_m.iloc[-1])))


# ----------------------------------------------------------------- (b) output gap
def output_gaps(ctx):
    P, R = ctx.P, ctx.R
    y = (100 * P["sa_lrgdp"]).loc["2004-04-01":"2026-04-01"].dropna()
    t = np.arange(len(y))
    cyc_hp, tr_hp = hpfilter(y, lamb=1600)

    # Hamilton (2018): regress y_{t+8} on y_t..y_{t-3}; gap = 2-year-ahead forecast error
    H = pd.DataFrame({"y8": y})
    for j in range(4):
        H[f"l{j}"] = y.shift(8 + j)
    XH = sm.add_constant(H[[f"l{j}" for j in range(4)]])
    hres = sm.OLS(H.y8, XH, missing="drop").fit()
    gap_ham = (H.y8 - hres.predict(XH)).reindex(y.index)
    # robustness: drop any observation whose LHS or regressors touch 2020Q2-2021Q2
    cov_s = pd.Series(((y.index >= COVID_START) & (y.index <= COVID_END)).astype(float), index=y.index)
    touch = cov_s.copy()
    for j in range(8, 12):
        touch = touch + cov_s.shift(j).fillna(0)
    Hx = H[touch == 0]
    hres_x = sm.OLS(Hx.y8, sm.add_constant(Hx[[f"l{j}" for j in range(4)]]), missing="drop").fit()
    gap_ham_x = (H.y8 - hres_x.predict(XH)).reindex(y.index)

    # deterministic linear trend with a 2020Q2 level break; plain linear trend
    Xtr = sm.add_constant(np.column_stack([t, (y.index >= COVID_START).astype(float)]))
    tres = sm.OLS(y.values, Xtr).fit()
    Xl = sm.add_constant(t)
    G = pd.DataFrame({
        "HP (standard)": cyc_hp, "Hamilton": gap_ham, "Hamilton (ex-COVID est.)": gap_ham_x,
        "Trend+break": pd.Series(y.values - tres.predict(Xtr), index=y.index),
        "Linear trend": pd.Series(y.values - sm.OLS(y.values, Xl).fit().predict(Xl), index=y.index)})

    # Baseline HP: pandemic quarters interpolated before filtering, gap vs actual output
    y_int = y.copy(); y_int.loc[COVID_START:COVID_END] = np.nan; y_int = y_int.interpolate()
    _, tr_int = hpfilter(y_int, lamb=1600); G["HP"] = y - tr_int
    _, tr400 = hpfilter(y_int, lamb=400); G["HP (λ=400)"] = y - tr400
    _, t11 = hpfilter(y_int.loc["2011-04-01":], lamb=1600); G["HP (from 2011)"] = y - t11
    _, t25 = hpfilter(y_int.loc[:"2025-10-01"], lamb=1600); G["HP (end 2025Q4)"] = (y - t25).loc[:"2025-10-01"]
    rt = {}
    for end in y.loc["2018-01-01":].index:  # one-sided (real-time) HP
        c, _ = hpfilter(y.loc[:end], lamb=1600); rt[end] = c.iloc[-1]
    G["HP real-time"] = pd.Series(rt)
    ctx.P = P = P.join(G)

    fig, ax = plt.subplots(2, 1, figsize=(7.2, 5.8))
    ax[0].plot(y.index, y, color=INK, lw=1.2, label="log real GDP (SA, ×100)")
    ax[0].plot(y.index, tr_int, color=C1, label="HP trend (λ=1600, pandemic interpolated)")
    ax[0].plot(y.index, tr_hp, color=C1, ls=":", lw=1, label="HP trend, standard")
    ax[0].plot(y.index, tres.predict(Xtr), color=C2, ls="--", label="Linear trend + 2020Q2 break")
    ax[0].legend(fontsize=7); ax[0].set_ylabel("log points")
    zz = G.loc["2014-01-01":]
    for c, st, col in [("HP", "-", C1), ("Hamilton", "-", C2), ("Trend+break", "--", C3), ("HP real-time", ":", C4)]:
        ax[1].plot(zz.index, zz[c], ls=st, lw=1.4, color=col, label=c)
    ax[1].axhline(0, color=INK, lw=.6); ax[1].set_ylim(-14, 12)
    ax[1].legend(fontsize=7, ncol=4); ax[1].set_ylabel("Output gap (% of potential)")
    save(fig, "fig2_output_gap")

    ep = {"2019Q4 (pre-COVID)": ["2019-10-01"], "2020Q2 (trough)": ["2020-04-01"],
          "FY2021-22 avg": ["2021-04-01", "2022-01-01"], "FY2022-23 avg": ["2022-04-01", "2023-01-01"],
          "2025 avg": ["2025-01-01", "2025-10-01"], "2026Q2 (latest)": ["2026-04-01"]}
    gcols = ["HP", "HP (standard)", "Hamilton", "Hamilton (ex-COVID est.)", "Trend+break", "Linear trend",
             "HP (λ=400)", "HP (from 2011)", "HP (end 2025Q4)", "HP real-time"]
    gt = pd.DataFrame({k: G.loc[v[0]:v[-1], gcols].mean() for k, v in ep.items()}).T
    gt.index.name = "Episode"
    frame_table(gt, "t02_output_gap", "Table 2. Output gap by episode (% of potential): method, smoothing, sample and end-point sensitivity.")
    R["gaps_episode"] = gt.round(2).to_dict()
    R["gap_corr"] = G[["HP", "Hamilton", "Trend+break"]].loc["2011-04-01":].corr().round(2).to_dict()
    R["hamilton"] = dict(coefs=hres.params.round(3).to_dict(), coefs_exCOVID=hres_x.params.round(3).to_dict(),
                         n_exCOVID=int(hres_x.nobs))
    R["trend_break"] = dict(slope_annual=float(4 * tres.params[1]), level_break=float(tres.params[2]))


# ----------------------------------------------------------------- (c) inflation equation
def inflation_equation(ctx):
    P, R = ctx.P, ctx.R
    E = P.loc["2014-01-01":"2026-04-01", ["pi", "HP", "Hamilton", "Trend+break"]].copy()
    E["pi_l1"] = E.pi.shift(1)
    for g in ["HP", "Hamilton", "Trend+break"]:
        E[g + "_l1"] = E[g].shift(1)
    E = E.dropna()

    def fit(gap, data=E, lagged=True):
        return ols(data.pi, data[["pi_l1", gap] + ([gap + "_l1"] if lagged else [])], hac=HAC_Q)

    m0 = fit("HP", lagged=False); m1 = fit("HP"); m2 = fit("Hamilton"); m3 = fit("Trend+break")
    m4 = fit("HP", E.drop(E.loc[COVID_START:COVID_END].index))
    out = {}
    for nm, m, g in [("HP", m1, "HP"), ("Ham", m2, "Hamilton"), ("TB", m3, "Trend+break"), ("HPexCOVID", m4, "HP")]:
        wt = m.wald_test(f"{g} = 0, {g}_l1 = 0", use_f=True, scalar=True)
        comb = lincomb(m, {g: 1, g + "_l1": 1})
        r = m.params["pi_l1"]
        V = m.cov_params().loc[[g, g + "_l1", "pi_l1"], [g, g + "_l1", "pi_l1"]].values
        grad = np.array([1 / (1 - r), 1 / (1 - r), comb["est"] / (1 - r) ** 2])
        out[nm] = dict(F=float(wt.statistic), pF=float(wt.pvalue), comb=comb, rho=float(r),
                       lr=comb["est"] / (1 - r), lr_se=float(np.sqrt(grad @ V @ grad)),
                       bg_p=bg_pvalue(m, 4), n=int(m.nobs), params=m.params.to_dict(), bse=m.bse.to_dict())
    out["lag_adds"] = dict(p_lag=float(m1.pvalues["HP_l1"]), aic_nolag=float(m0.aic), aic_lag=float(m1.aic),
                           r2a_nolag=float(m0.rsquared_adj), r2a_lag=float(m1.rsquared_adj))
    out["half_life_q"] = float(np.log(0.5) / np.log(out["HP"]["rho"]))
    R["inflation"] = out
    ks = ["HP", "Ham", "TB", "HPexCOVID"]
    reg_table([m0, m1, m2, m3, m4], ["(1) HP, no lag", "(2) HP", "(3) Hamilton", "(4) Trend+brk", "(5) HP ex-COVID"],
              [("const", "Constant"), ("pi_l1", "π(t-1)"), ("HP", "gap(t)"), ("HP_l1", "gap(t-1)"),
               ("Hamilton", "gap(t), Hamilton"), ("Hamilton_l1", "gap(t-1), Hamilton"),
               ("Trend+break", "gap(t), trend"), ("Trend+break_l1", "gap(t-1), trend")],
              "t03_inflation", "Table 3. Inflation equations. Dependent variable: π(t), quarterly CPI-C y/y. HAC(4) s.e.",
              extra=[("Joint F: gap terms = 0 (p)", ["–"] + [f"{out[k]['pF']:.3f}" for k in ks]),
                     ("β0 + β1", [f"{m0.params['HP']:.3f}"] + [f"{out[k]['comb']['est']:.3f}" for k in ks]),
                     ("Breusch–Godfrey(4) p", [f"{bg_pvalue(m0, 4):.3f}"] + [f"{out[k]['bg_p']:.3f}" for k in ks])])


def run(ctx):
    indicators(ctx)
    output_gaps(ctx)
    inflation_equation(ctx)
    return ctx
