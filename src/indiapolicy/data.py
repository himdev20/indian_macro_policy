"""Load the raw, source-documented CSV files in data/raw/.

Every number used in the analysis lives in one of these files, with its source
and status (actual / provisional / RE / BE) recorded next to it. See
data/DATA_LOG.md for definitions.
"""
from dataclasses import dataclass
import pandas as pd

from .config import RAW


@dataclass
class RawData:
    Q: pd.DataFrame            # quarterly real GDP, nominal GDP, real PFCE (INR mn, NSA)
    NAS: dict                  # annual benchmarks used to splice the base-year break
    cpi_yoy: pd.Series         # monthly CPI-C inflation, % y/y
    repo_changes: pd.Series    # policy repo rate by effective date
    rates_A: pd.DataFrame      # end-March lending/deposit rates
    rates_M: pd.DataFrame      # monthly lending/deposit rates, 2025-26
    nonfood: pd.Series         # non-food bank credit, Rs crore
    nonfood_exmerger: pd.Series
    FISC: pd.DataFrame         # Centre's finances, Rs crore


def load_raw(raw_dir=RAW) -> RawData:
    q = pd.read_csv(raw_dir / "quarterly_national_accounts.csv", index_col=0, parse_dates=True)
    Q = q[["real_gdp_inr_mn", "nominal_gdp_inr_mn", "real_pfce_inr_mn"]].set_axis(["rgdp", "ngdp", "rpfce"], axis=1)
    Q.index.name = None

    n = pd.read_csv(raw_dir / "nas_annual_benchmarks.csv")
    def v(series, base, fy):
        return int(n.query("series == @series and base == @base and fiscal_year_ending == @fy").value.iloc[0])
    def d(series, base):
        s = n.query("series == @series and base == @base")
        return {int(r.fiscal_year_ending): int(r.value) for r in s.itertuples()}
    NAS = dict(
        rgdp_old_FY22=v("real_gdp", "2011-12", 2022), rgdp_old_FY23=v("real_gdp", "2011-12", 2023),
        rgdp_new_FY23=v("real_gdp", "2022-23", 2023), rgdp_new=d("real_gdp", "2022-23"),
        ngdp_old=d("nominal_gdp", "2011-12"), ngdp_new=d("nominal_gdp", "2022-23"),
        pfce_pc_old_FY22=v("real_pfce_per_capita", "2011-12", 2022),
        pfce_pc_old_FY23=v("real_pfce_per_capita", "2011-12", 2023),
        pfce_pc_new_FY23=v("real_pfce_per_capita", "2022-23", 2023),
        pop_FY22_lakh=v("population", "2011-12", 2022),
        gndi_cur=d("gndi_current", "2022-23"),
    )

    c = pd.read_csv(raw_dir / "cpi_inflation_monthly.csv")
    cpi = pd.Series(c.cpi_c_yoy_pct.values, index=pd.to_datetime(c.month + "-01")).sort_index()

    r = pd.read_csv(raw_dir / "repo_rate_changes.csv")
    repo = pd.Series(r.repo_rate_pct.values, index=pd.to_datetime(r.effective_date))

    A = pd.read_csv(raw_dir / "bank_rates_annual.csv", index_col=0)
    A = A[["walr_out", "walr_fresh", "mclr_1y", "wadtdr_out", "call", "repo"]]
    A.index.name = None

    M = pd.read_csv(raw_dir / "bank_rates_monthly.csv", index_col=0)
    M = M[["walr_fresh", "walr_out", "wadtdr_fresh", "wadtdr_out"]]
    M.index.name = None

    cr = pd.read_csv(raw_dir / "nonfood_bank_credit.csv", index_col=0)

    f = pd.read_csv(raw_dir / "central_government_finances.csv", index_col=0)
    FISC = f.iloc[:, :9].set_axis(["GFD", "GPD", "RD", "REVEX", "INT", "CAPEX", "PIT", "DIR", "IND"], axis=1)
    FISC.index.name = None

    return RawData(Q=Q, NAS=NAS, cpi_yoy=cpi, repo_changes=repo, rates_A=A, rates_M=M,
                   nonfood=cr.nonfood_credit_rs_crore.dropna(),
                   nonfood_exmerger=cr.nonfood_credit_ex_merger_rs_crore.dropna(), FISC=FISC)
