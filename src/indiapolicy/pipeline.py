"""End-to-end pipeline: raw CSVs -> processed datasets, figures, tables, results.json."""
import json
import time
import warnings
from types import SimpleNamespace

import numpy as np

from . import q1_macro_position, q2_monetary_policy, q3_transmission, q4_fiscal_policy, q5_households, unit_roots
from .build import quarterly_panel, splice_quarterly
from .config import OUT, PROCESSED
from .data import load_raw

STEPS = [("Q1  macro position & output gap", q1_macro_position),
         ("Q2  Taylor rule & reaction function", q2_monetary_policy),
         ("Q3  transmission to bank rates", q3_transmission),
         ("Q4  fiscal stance & expenditure", q4_fiscal_policy),
         ("Q5  households & policy mix", q5_households),
         ("     unit-root diagnostics", unit_roots)]


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    return o


def build_context():
    D = load_raw()
    P, cpi_m, repo_d = quarterly_panel(D)
    FYr = splice_quarterly(D).groupby("fy")[["rgdp", "ngdp", "rpfce"]].sum()  # fiscal-year totals, INR mn
    R = {"splice": {k: round(float(v), 4) for k, v in P.attrs["splice_factors"].items()}}
    return SimpleNamespace(D=D, P=P, cpi_m=cpi_m, repo_d=repo_d, FYr=FYr, gFY=100 * FYr.rgdp.pct_change(), R=R)


def run(verbose=True):
    warnings.filterwarnings("ignore")
    ctx = build_context()
    for label, mod in STEPS:
        t0 = time.time()
        mod.run(ctx)
        if verbose:
            print(f"  {label:<40s} {time.time() - t0:5.1f}s")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(_jsonable(ctx.R), indent=1, default=str))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    ctx.P.drop(columns=["lrgdp", "lngdp", "lrpfce"]).to_csv(PROCESSED / "quarterly_panel.csv", index_label="quarter_start")
    ctx.FI.to_csv(PROCESSED / "annual_fiscal.csv", index_label="fiscal_year_ending")
    ctx.D.rates_A.to_csv(PROCESSED / "annual_bank_rates.csv", index_label="fiscal_year_ending")
    return ctx
