# Data log

Every number used in the analysis is stored in `data/raw/` with its source and
status next to it. Nothing is downloaded at run time, so results are fully
reproducible offline. All sources were accessed on 30 October 2026.

## Raw files

| File | Variables | Units / frequency | Source | Status / notes |
|---|---|---|---|---|
| `quarterly_national_accounts.csv` | Real GDP, nominal GDP, real private final consumption expenditure (PFCE) | INR million, not seasonally adjusted; quarterly 2004Q2–2026Q2 | IMF *International Financial Statistics* via FRED: `NGDPRNSAXDCINQ`, `NGDPNSAXDCINQ`, `NCPRNSAXDCINQ` (updated 7 Sep 2026) | New MoSPI base (2022-23) from 2022Q2; earlier quarters are an IMF-linked back series. FY2025-26 quarters provisional; 2026Q2 first estimate. |
| `nas_annual_benchmarks.csv` | Annual real/nominal GDP under both bases, real PFCE per capita, population, GNDI | Rs crore (PFCE per capita: Rs; population: lakh) | RBI *Handbook of Statistics on the Indian Economy 2025-26*, Tables 1–2 (NSO) | 2022-23 published under both bases, which identifies the splice ratio. 2024-25 first revised; 2025-26 provisional. |
| `cpi_inflation_monthly.csv` | CPI-Combined inflation | % y/y, monthly Jan 2014–Sep 2026 | MoSPI/PIB release 12 Feb 2025 (Annex V–VI); RBI Handbook Table 243; RBI MPC resolution 7 Oct 2026; MoSPI Sep 2026 release | Base 2012=100 to Dec 2025, 2024=100 from Jan 2026 (published y/y rates on each base). Apr–May 2020 not released: computed from imputed indices. Sep 2026 provisional. |
| `repo_rate_changes.csv` | Policy repo rate | %, effective dates | RBI Handbook Table 40; RBI MPC resolution 7 Oct 2026 | Converted to a daily step function, then quarter averages / ends. |
| `bank_rates_annual.csv` | WALR (fresh, outstanding), 1-yr median MCLR, WADTDR (outstanding), call rate, repo | %, end-March FY2013-14 to FY2025-26, plus May 2026 (labelled 2027) | RBI Handbook Table 59 (repo: Table 40) | Scheduled commercial banks. |
| `bank_rates_monthly.csv` | WALR (fresh, outstanding), WADTDR (fresh, outstanding) | %, monthly Feb 2025–Aug 2026 | RBI press releases, *Lending and Deposit Rates of Scheduled Commercial Banks* | Months not retrieved are blank. |
| `nonfood_bank_credit.csv` | Non-food bank credit | Rs crore, end-March 2013–2026 | RBI Handbook Table 41 | FY24/FY25 also published excluding the July 2023 HDFC merger; growth uses the ex-merger values. |
| `central_government_finances.csv` | Gross fiscal / primary / revenue deficit; revenue & capital expenditure; interest; personal income tax; direct & indirect taxes (Centre's net share) | Rs crore, FY2004-05 to FY2026-27 | RBI Handbook Tables 86, 87, 89 (Union Budget documents) | Actuals to 2024-25; **RE** for 2025-26; **BE** for 2026-27 (nominal GDP = 1.10 × 2025-26, the Budget assumption). |

## Constructed variables (see `src/indiapolicy/build.py`)

| Variable | Definition |
|---|---|
| Spliced real GDP / PFCE / nominal GDP | Old-base quarters rescaled so FY2021-22 = official old-base value × (new/old 2022-23 ratio). Spliced 2022-23 annual growth is 7.85% vs the official 7.6% (0.25 pp; affects the level only, not the cycle). |
| Seasonal adjustment | Old-base segment: robust STL (period 4, seasonal window 7). New-base segment (17 quarters): OLS fixed quarter effects with a linear trend. |
| `y_t` | 100 × log SA real GDP. Growth `g_t` = four-quarter change in `y_t`. |
| `π_t` | Quarter average of monthly CPI-C y/y inflation. |
| `i_t` | Quarter average of the daily repo rate. |
| Output gap `ỹ_t` | % of potential; HP (λ=1600, pandemic quarters interpolated before filtering), Hamilton (2018) regression filter, linear trend with a 2020Q2 level break, and several sensitivity variants. |
| Fiscal ratios | % of nominal GDP on the official basis (old base to FY22, new base from FY23). |
| CAPB, fiscal impulse | `CAPB = PB − ε·ỹ` with ε = 0.10 (0.20 sensitivity); `FI = −ΔCAPB` (+ = expansionary). |
| `Δk_t`, `Δr_t` | Change in Centre capital / revenue-ex-interest expenditure as % of last year's nominal GDP. |

## Status flags

Actual observations, provisional estimates, revised estimates (RE) and budget
estimates (BE) are flagged in the `status` columns. The only forward-looking
numbers in the analysis are the 2026-27 BE fiscal figures, which are labelled
wherever they appear.
