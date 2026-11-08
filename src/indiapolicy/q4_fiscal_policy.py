"""Q4. Fiscal policy: cyclically adjusted primary balance, fiscal impulse and
expenditure-growth regressions."""
import pandas as pd
import matplotlib.pyplot as plt

from .config import FISCAL_EPS, FISCAL_EPS_ALT, NOMINAL_GROWTH_BE_2027, HAC_A, INK, C1, C2
from .econometrics import ols, lincomb, bg_pvalue
from .plotting import save, fy_ticks
from .tables import frame_table, reg_table


# ----------------------------------------------------------------- (a) fiscal stance
def fiscal_stance(ctx):
    D, P, R = ctx.D, ctx.P, ctx.R
    FI = D.FISC.copy().astype(float)
    # Nominal GDP on the official basis: old base to FY22, new base from FY23; FY27 = BE assumption
    fyN = ctx.FYr.ngdp / 10                       # spliced (new-base level), Rs crore
    kN = ctx.R["splice"]["k_nom"]                  # new/old nominal level ratio (4 d.p., as in the report)
    ngdp_off = (fyN / kN).loc[2005:2022].to_dict()
    ngdp_off.update(D.NAS["ngdp_new"])
    ngdp_off[2027] = D.NAS["ngdp_new"][2026] * (1 + NOMINAL_GROWTH_BE_2027)
    ngdp_link = fyN.loc[2005:2026].to_dict(); ngdp_link.update(D.NAS["ngdp_new"]); ngdp_link[2027] = ngdp_off[2027]
    FI["Y"] = pd.Series(ngdp_off); FI["Ylink"] = pd.Series(ngdp_link)
    for c in ["GFD", "GPD", "RD", "REVEX", "CAPEX", "INT", "PIT", "DIR", "IND"]:
        FI[c + "_y"] = 100 * FI[c] / FI.Y
    FI["PB_y"] = -FI.GPD_y                                     # surplus positive
    FI["gap"] = P["HP"].groupby(P.fy).mean()                   # FY mean of quarterly HP gap
    FI["gap_ham"] = P["Hamilton"].groupby(P.fy).mean()
    for eps, tag in [(FISCAL_EPS, ""), (FISCAL_EPS_ALT, "_e2")]:
        FI["CAPB" + tag] = FI.PB_y - eps * FI.gap
        FI["FI" + tag] = -FI["CAPB" + tag].diff()              # + = expansionary
    FI["CAPB_ham"] = FI.PB_y - FISCAL_EPS * FI.gap_ham; FI["FI_ham"] = -FI.CAPB_ham.diff()
    FI["dgap"] = FI.gap.diff()
    ctx.FI = FI

    ft = FI.loc[2015:2027, ["GFD_y", "GPD_y", "RD_y", "CAPEX_y", "REVEX_y", "gap", "CAPB", "FI", "FI_e2"]].copy()
    ft.index = [f"{y-1}-{str(y)[2:]}" + ("RE" if y == 2026 else "BE" if y == 2027 else "") for y in ft.index]
    ft.index.name = "FY"
    ft.columns = ["GFD", "GPD", "RD", "CapEx", "RevEx", "Gap", "CAPB", "FI(ε=.1)", "FI(ε=.2)"]
    frame_table(ft, "t10_fiscal", "Table 10. Central government (% of GDP), output gap, CAPB and fiscal impulse (pp of GDP; + = expansionary).")
    R["fiscal"] = FI.loc[2015:2027, ["GFD_y", "GPD_y", "RD_y", "CAPEX_y", "REVEX_y", "gap", "CAPB", "FI", "FI_e2", "FI_ham"]].round(2).to_dict()
    cyc = FI.loc[2016:2027]
    R["fiscal_cyclicality"] = dict(corr_FI_gap=float(cyc.FI.corr(cyc.gap)), corr_FI_dgap=float(cyc.FI.corr(cyc.dgap)))

    xx = FI.loc[2015:2027]
    fig, ax = plt.subplots(1, 2, figsize=(7.4, 3.3))
    ax[0].bar(xx.index - 0.2, xx.CAPEX_y, 0.38, color=C1, label="Capital exp.")
    ax[0].bar(xx.index + 0.2, xx.REVEX_y - xx.INT_y, 0.38, color=C2, label="Revenue exp. ex-interest")
    ax[0].plot(xx.index, xx.GFD_y, "-o", color=INK, ms=3, label="Gross fiscal deficit")
    ax[0].set_ylabel("% of GDP"); ax[0].legend(fontsize=6.5); ax[0].set_title("Centre: composition", fontsize=8)
    ax[1].bar(xx.index, xx.FI, color=[C2 if v > 0 else C1 for v in xx.FI], label="Fiscal impulse (+ = expansionary)")
    ax[1].plot(xx.index, xx.gap, "-o", color=INK, ms=3, label="Output gap (HP, FY avg)")
    ax[1].axhline(0, color=INK, lw=.6); ax[1].set_ylim(-13, 3.6); ax[1].legend(fontsize=6.5, loc="lower right"); ax[1].set_title("Fiscal impulse vs cycle", fontsize=8)
    for a in ax:
        fy_ticks(a, range(2015, 2028, 2))
    save(fig, "fig5_fiscal")


# ----------------------------------------------------------------- (b,c) expenditure and growth
def expenditure_growth(ctx):
    FI, gFY, R = ctx.FI, ctx.gFY, ctx.R
    X = pd.DataFrame(index=range(2006, 2027))
    X["g"] = gFY; X["g_l1"] = gFY.shift(1)
    YL = FI.Ylink
    X["dk"] = 100 * (FI.CAPEX - FI.CAPEX.shift(1)) / YL.shift(1)                       # pp of lagged GDP
    X["dr"] = 100 * ((FI.REVEX - FI.INT) - (FI.REVEX - FI.INT).shift(1)) / YL.shift(1)  # ex-interest
    X["dk_l1"] = X.dk.shift(1); X["dr_l1"] = X.dr.shift(1)
    X["D21"] = (X.index == 2021).astype(float); X["D22"] = (X.index == 2022).astype(float)
    X = X.loc[2007:2026]
    ctx.X4 = X
    f0 = ols(X.g, X[["g_l1", "dk", "dr"]], hac=HAC_A)
    f1 = ols(X.g, X[["g_l1", "dk", "dr", "D21", "D22"]], hac=HAC_A)
    f2 = ols(X.g, X[["g_l1", "dk", "dr", "dk_l1", "dr_l1", "D21", "D22"]].dropna(), hac=HAC_A)
    f3 = ols(X.g, X[["g_l1", "dk_l1", "dr_l1", "D21", "D22"]].dropna(), hac=HAC_A)
    R["expenditure"] = dict(
        b=dict(bK=float(f1.params.dk), seK=float(f1.bse.dk), bR=float(f1.params.dr), seR=float(f1.bse.dr),
               p_equal=float(f1.wald_test("dk = dr", use_f=True, scalar=True).pvalue), n=int(f1.nobs)),
        c=dict(cumK=lincomb(f2, {"dk": 1, "dk_l1": 1}), cumR=lincomb(f2, {"dr": 1, "dr_l1": 1}),
               p_equal_cum=float(f2.wald_test("dk + dk_l1 = dr + dr_l1", use_f=True, scalar=True).pvalue),
               lagK=float(f2.params.dk_l1), lagK_se=float(f2.bse.dk_l1), bg_p=bg_pvalue(f2, 1)),
        sd_dk=float(X.dk.std()), sd_dr=float(X.dr.std()))
    E = R["expenditure"]
    reg_table([f0, f1, f2, f3], ["(1)", "(2)", "(3) + lags", "(4) lags only"],
              [("const", "Constant"), ("g_l1", "g(t-1)"), ("dk", "Δk(t)"), ("dr", "Δr(t)"), ("dk_l1", "Δk(t-1)"),
               ("dr_l1", "Δr(t-1)"), ("D21", "FY2020-21 dummy"), ("D22", "FY2021-22 dummy")],
              "t11_expenditure_growth", "Table 11. Expenditure–growth regressions. Dependent variable: real GDP growth (%). HAC(1) s.e.",
              extra=[("Wald βK = βR (p)", [f"{f0.wald_test('dk = dr', use_f=True, scalar=True).pvalue:.3f}",
                                           f"{E['b']['p_equal']:.3f}", f"{E['c']['p_equal_cum']:.3f}", "–"]),
                     ("Cum. capital (ΣβK)", ["–", "–", f"{E['c']['cumK']['est']:.2f} ({E['c']['cumK']['se']:.2f})", "–"]),
                     ("Cum. revenue (ΣβR)", ["–", "–", f"{E['c']['cumR']['est']:.2f} ({E['c']['cumR']['se']:.2f})", "–"]),
                     ("Breusch–Godfrey(1) p", [f"{bg_pvalue(m, 1):.3f}" for m in (f0, f1, f2, f3)])])


def run(ctx):
    fiscal_stance(ctx)
    expenditure_growth(ctx)
    return ctx
