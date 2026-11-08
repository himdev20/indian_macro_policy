"""Q3. Transmission of repo-rate changes to bank lending and deposit rates, and
the lead-lag relation between lending rates and credit growth."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.stats.stattools import durbin_watson

from .config import HAC_A, INK, C1, C2, C3, C4
from .econometrics import ols, lincomb, bg_pvalue
from .plotting import save
from .tables import frame_table, reg_table

RATES = ["walr_fresh", "walr_out", "mclr_1y", "wadtdr_out"]
NAMES = ["WALR fresh", "WALR outst.", "1-yr MCLR", "WADTDR outst."]


# ----------------------------------------------------------------- (a) episode pass-through
def episode_passthrough(ctx):
    A, R, repo_d = ctx.D.rates_A, ctx.R, ctx.repo_d

    def pt(a, b):
        di = A.loc[b, "repo"] - A.loc[a, "repo"]
        res = {v: dict(chg_bp=100 * (A.loc[b, v] - A.loc[a, v]), ratio=(A.loc[b, v] - A.loc[a, v]) / di) for v in RATES}
        return res | {"repo_bp": 100 * di}

    W = {"T1: Mar-22→Mar-23": pt(2022, 2023), "T2: Mar-22→Mar-24": pt(2022, 2024),
         "E1: Mar-25→Mar-26": pt(2025, 2026), "E2: Mar-24→May-26": pt(2024, 2027)}
    cols = ["Δrepo", "ΔWALR-F", "PT (WALR-F)", "ΔWALR-O", "PT (WALR-O)", "ΔMCLR", "PT (MCLR)", "ΔWADTDR-O", "PT (WADTDR-O)"]
    tab = pd.DataFrame({k: [v["repo_bp"]] + sum([[v[x]["chg_bp"], v[x]["ratio"]] for x in RATES], [])
                        for k, v in W.items()}, index=cols).T

    # monthly-release window; the policy change is measured over each series' own window
    M = ctx.D.rates_M.copy(); M.index = pd.PeriodIndex(M.index, freq="M"); M = M.sort_index()
    mw = {"walr_fresh": ("2025-02", "2026-08"), "walr_out": ("2025-04", "2026-08"),
          "wadtdr_fresh": ("2025-02", "2026-07"), "wadtdr_out": ("2025-02", "2026-08")}

    def drepo(a, b):
        return float(repo_d.loc[:pd.Period(b).end_time].iloc[-1] - repo_d.loc[:pd.Period(a).start_time].iloc[-1])
    pm = {v: dict(a=a, b=b, di=drepo(a, b), chg_bp=100 * (M.loc[b, v] - M.loc[a, v]),
                  ratio=(M.loc[b, v] - M.loc[a, v]) / drepo(a, b)) for v, (a, b) in mw.items()}
    tab.loc["E3: Feb-25→Aug-26‡"] = [100 * pm["walr_fresh"]["di"], pm["walr_fresh"]["chg_bp"], pm["walr_fresh"]["ratio"],
                                     pm["walr_out"]["chg_bp"], pm["walr_out"]["ratio"], np.nan, np.nan,
                                     pm["wadtdr_out"]["chg_bp"], pm["wadtdr_out"]["ratio"]]
    tab.index.name = "Window"
    frame_table(tab, "t07_passthrough", "Table 7. Changes (bp) and simple pass-through ratios ΔR/Δi by window. ‡Monthly releases; WALR-O window starts Apr-25 (Δi = −100 bp).")
    R["passthrough"] = tab.round(3).to_dict()
    R["passthrough_monthly"] = pm

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3))
    for col, lab, c in [("repo", "Repo", INK), ("walr_fresh", "WALR fresh", C1), ("walr_out", "WALR outstanding", C2),
                        ("wadtdr_out", "WADTDR outstanding", C3)]:
        ax[0].plot(A.index, A[col], "-o", ms=3, color=c, label=lab)
    ax[0].set_title("End-March, FY2013-14 to May-2026", fontsize=8); ax[0].legend(fontsize=6.5)
    ax[0].set_xticks(range(2014, 2028, 2))
    ax[0].set_xticklabels([f"{y-1}-{str(y)[2:]}" for y in range(2014, 2028, 2)], rotation=45, fontsize=7)
    mm = M.copy(); mm.index = mm.index.to_timestamp()
    rpm = repo_d.resample("MS").mean().loc["2025-01-01":"2026-09-01"]
    ax[1].step(rpm.index, rpm.values, where="post", color=INK, label="Repo (monthly avg)")
    for c, lab, col in [("walr_fresh", "WALR fresh", C1), ("walr_out", "WALR outstanding", C2),
                        ("wadtdr_fresh", "WADTDR fresh", C4), ("wadtdr_out", "WADTDR outstanding", C3)]:
        ax[1].plot(mm.index, mm[c], "o-", ms=3, color=col, label=lab)
    ax[1].set_title("Monthly releases, 2025-26 easing", fontsize=8); ax[1].legend(fontsize=6.5, loc="center right")
    ax[1].tick_params(axis="x", rotation=45, labelsize=7)
    save(fig, "fig4_bank_rates")


# ----------------------------------------------------------------- (b) distributed lags
def distributed_lags(ctx):
    A, R = ctx.D.rates_A, ctx.R
    dA = A.diff()
    dA["di_l1"] = dA.repo.shift(1)
    dA["di_neg"] = dA.repo.where(dA.repo < 0, 0.0); dA["di_pos"] = dA.repo.where(dA.repo > 0, 0.0)
    dA["di_neg_l1"] = dA.di_neg.shift(1); dA["di_pos_l1"] = dA.di_pos.shift(1)
    dl = {}
    for v in RATES:
        d = dA[[v, "repo", "di_l1", "di_pos", "di_neg", "di_pos_l1", "di_neg_l1"]].loc[:2026].dropna()
        m = ols(d[v], d[["repo", "di_l1"]])  # OLS s.e.: T = 11, DW shows no strong autocorrelation
        ma = ols(d[v], d[["di_pos", "di_pos_l1", "di_neg", "di_neg_l1"]])
        wt = ma.wald_test("di_pos + di_pos_l1 - di_neg - di_neg_l1 = 0", use_f=True, scalar=True)
        dl[v] = dict(m=m, cum=lincomb(m, {"repo": 1, "di_l1": 1}),
                     cpos=lincomb(ma, {"di_pos": 1, "di_pos_l1": 1}), cneg=lincomb(ma, {"di_neg": 1, "di_neg_l1": 1}),
                     asym_p=float(wt.pvalue), dw=float(durbin_watson(m.resid)))
    R["dist_lag"] = {v: dict(b0=float(x["m"].params.repo), se0=float(x["m"].bse.repo), b1=float(x["m"].params.di_l1),
                             se1=float(x["m"].bse.di_l1), cum=x["cum"], cum_tight=x["cpos"]["est"],
                             cum_ease=x["cneg"]["est"], asym_p=x["asym_p"], dw=x["dw"], n=int(x["m"].nobs))
                     for v, x in dl.items()}
    reg_table([dl[v]["m"] for v in RATES], NAMES,
              [("const", "Constant"), ("repo", "Δi(t)  (β0)"), ("di_l1", "Δi(t-1)  (β1)")],
              "t08_distributed_lag", "Table 8. Distributed-lag pass-through, annual end-March changes (pp on pp). OLS s.e.",
              extra=[("Cumulative β0+β1", [f"{dl[v]['cum']['est']:.2f}" for v in RATES]),
                     ("  95% CI", [f"[{dl[v]['cum']['lo']:.2f}, {dl[v]['cum']['hi']:.2f}]" for v in RATES]),
                     ("Cum. tightening", [f"{dl[v]['cpos']['est']:.2f}" for v in RATES]),
                     ("Cum. easing", [f"{dl[v]['cneg']['est']:.2f}" for v in RATES]),
                     ("Asymmetry test p", [f"{dl[v]['asym_p']:.3f}" for v in RATES]),
                     ("Durbin–Watson", [f"{dl[v]['dw']:.2f}" for v in RATES])])


# ----------------------------------------------------------------- (c) lending rates and credit
def credit(ctx):
    D, R = ctx.D, ctx.R
    A = D.rates_A
    cr = D.nonfood.astype(float)
    g_cr = 100 * cr.pct_change()
    # strip the July-2023 HDFC merger from FY24 and FY25 growth
    g_cr.loc[2024] = 100 * (D.nonfood_exmerger[2024] / D.nonfood[2023] - 1)
    g_cr.loc[2025] = 100 * (D.nonfood_exmerger[2025] / D.nonfood_exmerger[2024] - 1)
    C = pd.DataFrame({"gcr": g_cr, "dW": A.walr_fresh.diff(), "dWo": A.walr_out.diff(), "g": ctx.gFY}).loc[2016:2026]
    C["dW_l1"] = C.dW.shift(1); C["dWo_l1"] = C.dWo.shift(1)
    R["credit_xcorr"] = {f"corr(gcr_t, dWALR_t{k:+d})": float(C.gcr.corr(C.dW.shift(-k))) for k in [-1, 0, 1]}
    Cc = C.dropna(subset=["gcr", "dW", "dW_l1", "g"])
    cm = ols(Cc.gcr, Cc[["dW", "dW_l1", "g"]], hac=HAC_A)
    cm2 = ols(Cc.gcr, Cc[["dWo", "dWo_l1", "g"]].dropna(), hac=HAC_A)
    cm_ols = ols(Cc.gcr, Cc[["dW", "dW_l1", "g"]])
    R["credit_reg"] = dict(params=cm.params.to_dict(), se_hac=cm.bse.to_dict(), se_ols=cm_ols.bse.to_dict(),
                           n=int(cm.nobs), bg_p=[bg_pvalue(m, 1) for m in (cm, cm2)])
    R["credit_growth"] = {int(k): round(float(v), 2) for k, v in g_cr.loc[2016:].items()}
    ctx.g_cr = g_cr
    reg_table([cm, cm2], ["WALR fresh", "WALR outst."],
              [("const", "Constant"), ("dW", "ΔWALR(t)"), ("dW_l1", "ΔWALR(t-1)"), ("dWo", "ΔWALR-O(t)"),
               ("dWo_l1", "ΔWALR-O(t-1)"), ("g", "Real GDP growth(t)")],
              "t09_credit", "Table 9. Non-food credit growth and lending-rate changes (annual, FY2016-17 to FY2025-26). HAC(1) s.e.",
              extra=[("Breusch–Godfrey(1) p", [f"{p:.3f}" for p in R["credit_reg"]["bg_p"]]),
                     ("Durbin–Watson", [f"{durbin_watson(m.resid):.2f}" for m in (cm, cm2)])])


def run(ctx):
    episode_passthrough(ctx)
    distributed_lags(ctx)
    credit(ctx)
    return ctx
