"""ADF unit-root tests for every regression variable (report Table 15)."""
import pandas as pd

from .econometrics import adf
from .tables import frame_table


def run(ctx):
    P, A = ctx.P, ctx.D.rates_A
    series = [
        ("Inflation π", P.pi.loc["2014":], "Q", "2014Q1–2026Q3"),
        ("Output gap (HP)", P.HP.loc["2011":], "Q", "2011Q1–2026Q2"),
        ("Output gap (Hamilton)", P.Hamilton.loc["2011":], "Q", "2011Q1–2026Q2"),
        ("Output gap (Hamilton, ex-COVID est.)", P["Hamilton (ex-COVID est.)"].loc["2011":], "Q", "2011Q1–2026Q2"),
        ("Repo rate i", P.repo_avg.loc["2014":"2026-04-01"], "Q", "2014Q1–2026Q2"),
        ("Δi", P.repo_avg.diff().loc["2014":"2026-04-01"], "Q", "2014Q2–2026Q2"),
        ("ΔWALR fresh", A.walr_fresh.diff(), "A", "FY16–FY27"),
        ("ΔWALR outstanding", A.walr_out.diff(), "A", "FY15–FY27"),
        ("ΔWADTDR outstanding", A.wadtdr_out.diff(), "A", "FY15–FY27"),
        ("Credit growth", ctx.g_cr.loc[2016:2026], "A", "FY16–FY26"),
        ("Real GDP growth g (annual)", ctx.X4.g, "A", "FY07–FY26"),
        ("Δk", ctx.X4.dk, "A", "FY07–FY26"),
        ("Δr", ctx.X4.dr, "A", "FY07–FY26"),
        ("ΔC (q/q)", ctx.C5.c_qoq, "Q", "2005Q3–2026Q2"),
        ("ΔY (q/q)", ctx.C5.g_qoq, "Q", "2005Q3–2026Q2"),
        ("PFCE growth (annual)", ctx.T5.c, "A", "FY07–FY26"),
        ("Δ(PIT/Y)", ctx.T5.dpit, "A", "FY07–FY26"),
    ]
    rows = {}
    for name, s, f, smp in series:
        r = adf(s, f)
        rows[name] = dict(Freq=f, Sample=smp, **{"ADF stat": r["stat"], "p-value": r["p"], "Lags": r["lags"], "Obs.": r["nobs"]})
    t = pd.DataFrame(rows).T
    t[["ADF stat", "p-value"]] = t[["ADF stat", "p-value"]].astype(float)
    t.index.name = "Series"
    frame_table(t, "t15_adf", "Table 15. Augmented Dickey–Fuller tests (constant, no trend; H0: unit root). Quarterly: AIC lags; annual: one lag.", fmt="{:.3f}")
    ctx.R["adf"] = {k: float(v) for k, v in t["p-value"].items()}
    return ctx
