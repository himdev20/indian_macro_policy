"""Q2. Did monetary policy respond systematically? Taylor-rule benchmark and
estimated reaction functions."""
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import grangercausalitytests

from .config import TAYLOR_RULES, POLICY_EPISODES, PI_STAR, HAC_Q, INK, C1, C2, C3
from .econometrics import ols, lincomb, ratio_sum, bg_pvalue
from .plotting import shade_episodes, save
from .tables import frame_table, reg_table


def q(t):
    return f"{t.year}Q{(t.month - 1) // 3 + 1}"


# ----------------------------------------------------------------- (a) Taylor rule
def taylor_rule(ctx):
    P, R = ctx.P, ctx.R
    TR = P.loc["2014-01-01":"2026-07-01", ["pi", "repo_avg", "HP", "Hamilton", "Trend+break"]].copy()
    for c in ["HP", "Hamilton", "Trend+break"]:
        TR[c] = TR[c].ffill()  # 2026Q3: GDP not yet released, carry 2026Q2 gap forward

    def rule(rstar, tp, ty, gap):
        return rstar + TR.pi + tp * (TR.pi - PI_STAR) + ty * TR[gap]

    for k, c in TAYLOR_RULES.items():
        TR[k] = rule(**c)
        TR["gap" + k] = TR.repo_avg - TR[k]

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    ax.plot(TR.index, TR.repo_avg, color=INK, lw=2.2, label="Actual repo (quarter avg)")
    ax.plot(TR.index, TR.A, color=C1, label="Rule A: r*=1.5, θπ=0.5, θy=0.5, HP gap")
    ax.plot(TR.index, TR.B, color=C2, ls="--", label="Rule B: r*=1.0, θπ=1.0, θy=0.5, Hamilton gap")
    ax.plot(TR.index, TR.C, color=C3, ls=":", label="Rule C: r*=1.9, θπ=0.5, θy=1.0, trend gap")
    shade_episodes(ax, [("2022-04-01", "2023-03-31"), ("2025-01-01", "2025-12-31")])
    ax.set_ylim(-8, 16); ax.set_ylabel("%")
    ax.legend(fontsize=6.5, ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0))
    save(fig, "fig3_taylor_rule")

    cols = ["repo_avg", "pi", "HP", "A", "gapA", "B", "gapB", "C", "gapC"]
    tt = pd.DataFrame({k: [TR.loc[a:b, c].mean() for c in cols] for k, (a, b) in POLICY_EPISODES.items()},
                      index=["Repo", "π", "Gap(HP)", "i^A", "i−i^A", "i^B", "i−i^B", "i^C", "i−i^C"]).T
    tt.index.name = "Episode (means, %)"
    frame_table(tt, "t04_taylor", "Table 4. Actual vs rule-implied policy rates (means, %). Negative i−i^X = looser than the rule. †2026Q3 gap carried forward.")
    R["taylor"] = tt.round(2).to_dict()

    # one-at-a-time sensitivity around Rule A
    sens = {"Rule A (baseline)": (1.5, 0.5, 0.5, "HP"), "r* = 1.0": (1.0, 0.5, 0.5, "HP"),
            "r* = 1.9": (1.9, 0.5, 0.5, "HP"), "θπ = 1.0": (1.5, 1.0, 0.5, "HP"),
            "θy = 1.0": (1.5, 0.5, 1.0, "HP"), "Hamilton gap": (1.5, 0.5, 0.5, "Hamilton"),
            "Trend+break gap": (1.5, 0.5, 0.5, "Trend+break")}
    win = [("2022-04-01", "2023-01-01"), ("2025-01-01", "2025-10-01")]
    st = {}
    for lab, (r_, tp_, ty_, g_) in sens.items():
        ir = rule(r_, tp_, ty_, g_)
        st[lab] = sum([[ir.loc[a:b].mean(), (TR.repo_avg - ir).loc[a:b].mean()] for a, b in win], [])
    st = pd.DataFrame(st, index=["i^TR tightening", "i−i^TR tightening", "i^TR easing", "i−i^TR easing"]).T
    st.index.name = "Change from Rule A"
    frame_table(st, "t05_taylor_sensitivity", "Table 5. One-at-a-time sensitivity: Rule A with a single element changed (episode means, %).")
    R["taylor_sens"] = st.round(2).to_dict()
    ctx.TR = TR


# ----------------------------------------------------------------- (b,c) reaction functions
def reaction_functions(ctx):
    P, R = ctx.P, ctx.R
    F = P.loc["2014-01-01":"2026-04-01", ["repo_avg", "pi", "HP", "Hamilton"]].copy()
    F["i_l1"] = F.repo_avg.shift(1); F["pi_l1"] = F.pi.shift(1); F["HP_l1"] = F.HP.shift(1)
    F["pi_l2"] = F.pi.shift(2); F["HP_l2"] = F.HP.shift(2)  # keeps the common sample of the report
    F = F.dropna()
    rb = ols(F.repo_avg, F[["i_l1", "pi", "HP"]], hac=HAC_Q)
    rbH = ols(F.repo_avg, F[["i_l1", "pi", "Hamilton"]], hac=HAC_Q)
    rc1 = ols(F.repo_avg, F[["i_l1", "pi", "HP", "pi_l1", "HP_l1"]], hac=HAC_Q)
    rc2 = ols(F.repo_avg, F[["i_l1", "pi_l1", "HP_l1"]], hac=HAC_Q)
    Fx = F.drop(F.loc["2020-01-01":"2021-04-01"].index)
    rbx = ols(Fx.repo_avg, Fx[["i_l1", "pi", "HP"]], hac=HAC_Q)

    spec = {"base": (rb, ["pi"], ["HP"]), "ham": (rbH, ["pi"], ["Hamilton"]),
            "lags": (rc1, ["pi", "pi_l1"], ["HP", "HP_l1"]), "lagonly": (rc2, ["pi_l1"], ["HP_l1"]),
            "exCOVID": (rbx, ["pi"], ["HP"])}
    rf = {}
    for nm, (m, pv, gv) in spec.items():
        rf[nm] = dict(rho=float(m.params["i_l1"]), n=int(m.nobs), aic=float(m.aic), bic=float(m.bic),
                      r2a=float(m.rsquared_adj), bg_p=bg_pvalue(m, 4),
                      LRpi=ratio_sum(m, pv, "i_l1"), LRy=ratio_sum(m, gv, "i_l1"))
        rf[nm]["tp_t"] = (rf[nm]["LRpi"]["est"] - 1) / rf[nm]["LRpi"]["se"]  # H0: LR inflation response = 1
    tpw = lincomb(rb, {"pi": 1, "i_l1": 1})
    rf["taylor_principle_wald"] = dict(est=tpw["est"] - 1, se=tpw["se"], t=(tpw["est"] - 1) / tpw["se"])
    rf["F_lags_p"] = float(rc1.wald_test("pi_l1 = 0, HP_l1 = 0", use_f=True, scalar=True).pvalue)
    R["reaction"] = rf

    # dynamic simulation of the baseline rule over the 2025 easing, from the actual 2024Q4 repo
    sim, i_prev = {}, float(P.loc["2024-10-01", "repo_avg"])
    b = rb.params
    for d in pd.date_range("2025-01-01", "2026-04-01", freq="QS"):
        i_prev = float(b["const"] + b["i_l1"] * i_prev + b["pi"] * P.loc[d, "pi"] + b["HP"] * P.loc[d, "HP"])
        sim[q(d)] = dict(sim=i_prev, actual=float(P.loc[d, "repo_avg"]))
    R["reaction_sim2025"] = sim
    R["reaction_resid2025"] = {q(d): float(v) for d, v in rb.resid.loc["2025-01-01":"2025-10-01"].items()}

    ks = list(spec)
    reg_table([rb, rbH, rc1, rc2, rbx], ["(1) Baseline", "(2) Hamilton", "(3) +lags", "(4) lags only", "(5) ex-COVID"],
              [("const", "Constant"), ("i_l1", "i(t-1)"), ("pi", "π(t)"), ("pi_l1", "π(t-1)"),
               ("HP", "gap(t), HP"), ("HP_l1", "gap(t-1), HP"), ("Hamilton", "gap(t), Hamilton")],
              "t06_reaction_function", "Table 6. Monetary-policy reaction functions. Dependent variable: repo rate (quarter avg). HAC(4) s.e.; long-run responses by delta method.",
              extra=[("LR π response", [f"{rf[k]['LRpi']['est']:.2f}" for k in ks]),
                     ("  (s.e.)", [f"({rf[k]['LRpi']['se']:.2f})" for k in ks]),
                     ("LR gap response", [f"{rf[k]['LRy']['est']:.2f}" for k in ks]),
                     ("  (s.e.) ", [f"({rf[k]['LRy']['se']:.2f})" for k in ks]),
                     ("BIC", [f"{rf[k]['bic']:.1f}" for k in ks]),
                     ("Breusch–Godfrey(4) p", [f"{rf[k]['bg_p']:.3f}" for k in ks])])

    # lead-lag correlations and Granger tests
    dI = F.repo_avg.diff()
    R["leadlag"] = {f"k={k}": dict(pi=float(dI.corr(P.pi.shift(k).reindex(dI.index))),
                                   gap=float(dI.corr(P.HP.shift(k).reindex(dI.index)))) for k in [-2, -1, 0, 1, 2, 3]}
    gc = pd.DataFrame({"di": dI, "dpi": P.pi.diff().reindex(dI.index)}).dropna()
    R["granger"] = dict(
        pi_to_i=float(grangercausalitytests(gc[["di", "dpi"]], maxlag=2)[2][0]["ssr_ftest"][1]),
        i_to_pi=float(grangercausalitytests(gc[["dpi", "di"]], maxlag=2)[2][0]["ssr_ftest"][1]))


def run(ctx):
    taylor_rule(ctx)
    reaction_functions(ctx)
    return ctx
