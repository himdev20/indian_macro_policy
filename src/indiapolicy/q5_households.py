"""Q5. Household consumption response to income and taxes; inputs for the
2025 policy-mix assessment."""
import pandas as pd

from .config import HAC_Q, HAC_A
from .econometrics import ols, lincomb, bg_pvalue
from .tables import reg_table


# ----------------------------------------------------------------- (a) consumption and income
def consumption_income(ctx):
    P, R = ctx.P, ctx.R
    C = P.loc["2005-04-01":"2026-04-01", ["c_qoq", "g_qoq", "c_yoy", "g_yoy", "repo_avg"]].copy()
    C["g_l1"] = C.g_qoq.shift(1); C["drepo_l1"] = C.repo_avg.diff().shift(1)
    for d in ["2020-04-01", "2020-07-01", "2021-04-01", "2021-07-01"]:
        C["D" + d[:7]] = (C.index == d).astype(float)
    dums = [c for c in C.columns if c.startswith("D20")]
    Cr = C.loc["2014-01-01":].dropna(subset=["drepo_l1"])
    c1 = ols(C.c_qoq, C[["g_qoq", "g_l1"] + dums].dropna(), hac=HAC_Q)
    c2 = ols(Cr.c_qoq, Cr[["g_qoq", "g_l1", "drepo_l1"] + dums], hac=HAC_Q)
    C["gy_l4"] = C.g_yoy.shift(4)
    c3 = ols(C.c_yoy, C[["g_yoy", "gy_l4"]].dropna(), hac=HAC_Q)
    Cx = C.drop(C.loc["2020-01-01":"2021-10-01"].index)
    c4 = ols(Cx.c_qoq, Cx[["g_qoq", "g_l1"]].dropna(), hac=HAC_Q)
    ctx.C5 = C
    models = {"qoq": c1, "qoq_rate": c2, "exCOVID": c4, "yoy": c3}
    out = {nm: dict(b0=float(m.params.iloc[1]), se0=float(m.bse.iloc[1]), b1=float(m.params.iloc[2]),
                    se1=float(m.bse.iloc[2]), comb=lincomb(m, {m.params.index[1]: 1, m.params.index[2]: 1}),
                    n=int(m.nobs), bg_p=bg_pvalue(m, 4)) for nm, m in models.items()}
    calm = C.loc["2011":].drop(C.loc["2020":"2021"].index)
    out["sd_ratio_c_to_y"] = float(calm.c_qoq.std() / calm.g_qoq.std())
    reg_table(list(models.values()), ["(1) q/q", "(2) q/q + rate", "(3) q/q ex-COVID", "(4) y/y"],
              [("const", "Constant"), ("g_qoq", "ΔY(t)"), ("g_l1", "ΔY(t-1)"), ("drepo_l1", "Δi(t-1)"),
               ("g_yoy", "Δ4Y(t)"), ("gy_l4", "Δ4Y(t-4)")],
              "t12_consumption", "Table 12. Consumption–income regressions (SA real PFCE on real GDP, log changes). HAC(4) s.e.",
              extra=[("β0 + β1", [f"{out[k]['comb']['est']:.2f}" for k in models]),
                     ("  95% CI", [f"[{out[k]['comb']['lo']:.2f}, {out[k]['comb']['hi']:.2f}]" for k in models]),
                     ("COVID dummies", ["Yes", "Yes", "Sample excl.", "No"]),
                     ("Breusch–Godfrey(4) p", [f"{out[k]['bg_p']:.3f}" for k in models])])

    # annual levels: marginal propensity to consume out of real GDP (Rs lakh crore)
    A5 = pd.DataFrame({"C": ctx.FYr.rpfce / 1e6, "Y": ctx.FYr.rgdp / 1e6}).loc[2005:2026]
    A5["dC"] = A5.C.diff(); A5["dY"] = A5.Y.diff(); A5["dY_l1"] = A5.dY.shift(1)
    A5["D21"] = (A5.index == 2021).astype(float); A5["D22"] = (A5.index == 2022).astype(float)
    c5 = ols(A5.dC, A5[["dY", "dY_l1", "D21", "D22"]], hac=HAC_A)
    out["mpc"] = dict(b0=float(c5.params.dY), se0=float(c5.bse.dY), b1=float(c5.params.dY_l1), se1=float(c5.bse.dY_l1),
                      comb=lincomb(c5, {"dY": 1, "dY_l1": 1}), n=int(c5.nobs),
                      apc=float((A5.C / A5.Y).loc[2016:2026].mean()))
    R["consumption"] = out
    reg_table([c5], ["ΔC(t), Rs lakh crore"],
              [("const", "Constant"), ("dY", "ΔY(t)"), ("dY_l1", "ΔY(t-1)"), ("D21", "FY2020-21"), ("D22", "FY2021-22")],
              "t13_mpc", "Table 13. Annual levels: marginal propensity to consume out of real GDP, FY2006-07 to FY2025-26. HAC(1) s.e.",
              extra=[("β0 + β1", [f"{out['mpc']['comb']['est']:.2f} ({out['mpc']['comb']['se']:.2f})"]),
                     ("Average C/Y, FY16–26", [f"{out['mpc']['apc']:.2f}"])])


# ----------------------------------------------------------------- (b) taxes and consumption
def taxes_consumption(ctx):
    FI, FYr, R = ctx.FI, ctx.FYr, ctx.R
    T = pd.DataFrame(index=range(2006, 2027))
    T["c"] = 100 * FYr.rpfce.pct_change(); T["g"] = ctx.gFY
    T["dpit"] = (100 * FI.PIT / FI.Ylink).diff()
    T["dind"] = (100 * FI.IND / FI.Ylink).diff()
    T["ddir"] = (100 * FI.DIR / FI.Ylink).diff()
    T["dpit_l1"] = T.dpit.shift(1)
    T["D21"] = (T.index == 2021).astype(float); T["D22"] = (T.index == 2022).astype(float)
    T = T.loc[2007:2026]
    # narrative dummy: FY2019-20 (Rs 5 lakh rebate), FY2023-24 (new regime default, Rs 7 lakh rebate),
    # FY2025-26 (Rs 12 lakh threshold + GST 2.0)
    T["TAXCUT"] = T.index.isin([2020, 2024, 2026]).astype(float)
    ctx.T5 = T
    ms = [ols(T.c, T[["dpit", "dpit_l1", "D21", "D22"]], hac=HAC_A),
          ols(T.c, T[["dpit", "dpit_l1", "g", "D21", "D22"]], hac=HAC_A),
          ols(T.c, T[["dpit", "dind", "g", "D21", "D22"]], hac=HAC_A),
          ols(T.c, T[["ddir", "dind", "g", "D21", "D22"]], hac=HAC_A),
          ols(T.c, T[["dpit", "dpit_l1", "g", "D21", "D22", "TAXCUT"]], hac=HAC_A)]
    out = {f"m{i+1}": {k: dict(b=float(m.params[k]), se=float(m.bse[k]), p=float(m.pvalues[k]))
                       for k in m.params.index if k not in ["const", "D21", "D22"]} for i, m in enumerate(ms)}
    out["m2"]["cum_pit"] = lincomb(ms[1], {"dpit": 1, "dpit_l1": 1})
    out["pit_to_gdp"] = (100 * FI.PIT / FI.Y).loc[2022:2027].round(2).to_dict()
    out["ind_to_gdp"] = (100 * FI.IND / FI.Y).loc[2023:2027].round(2).to_dict()
    out["bg_p"] = [bg_pvalue(m, 1) for m in ms]
    R["tax"] = out
    reg_table(ms, ["(1)", "(2)", "(3)", "(4)", "(5)"],
              [("const", "Constant"), ("dpit", "Δ(PIT/Y)(t)"), ("dpit_l1", "Δ(PIT/Y)(t-1)"), ("ddir", "Δ(DIR/Y)(t)"),
               ("dind", "Δ(IND/Y)(t)"), ("g", "Real GDP growth(t)"), ("TAXCUT", "Tax-relief year dummy"),
               ("D21", "FY2020-21 dummy"), ("D22", "FY2021-22 dummy")],
              "t14_tax", "Table 14. Tax changes and real consumption growth (annual). HAC(1) s.e.",
              extra=[("Breusch–Godfrey(1) p", [f"{p:.3f}" for p in out["bg_p"]])])


# ----------------------------------------------------------------- (c) inputs for the policy-mix verdict
def policy_mix_summary(ctx):
    P, R = ctx.P, ctx.R
    w = slice("2025-01-01", "2025-10-01")
    R["policy_mix_2025"] = dict(
        gap_HP=float(P.loc[w, "HP"].mean()), gap_Hamilton=float(P.loc[w, "Hamilton"].mean()),
        gap_trend_break=float(P.loc[w, "Trend+break"].mean()), gap_linear=float(P.loc[w, "Linear trend"].mean()),
        repo_avg=R["taylor"]["Repo"]["Easing 2025Q1–2025Q4"], rule_A=R["taylor"]["i^A"]["Easing 2025Q1–2025Q4"],
        pt_fresh_E3=R["passthrough"]["PT (WALR-F)"]["E3: Feb-25→Aug-26‡"],
        fiscal_impulse_FY26=R["fiscal"]["FI"][2026], capex_FY26=R["fiscal"]["CAPEX_y"][2026],
        bK=R["expenditure"]["b"]["bK"])


def run(ctx):
    consumption_income(ctx)
    taxes_consumption(ctx)
    policy_mix_summary(ctx)
    return ctx
