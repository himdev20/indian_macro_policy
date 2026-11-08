"""Replication tests: the pipeline must reproduce the headline numbers printed in
the submitted report (report/ECO612_report.pdf) to the reported precision."""
import pytest

from indiapolicy.pipeline import run


@pytest.fixture(scope="module")
def R():
    return run(verbose=False).R


def close(x, target, dp):
    return abs(x - target) <= 0.5 * 10 ** (-dp) + 1e-9


# ---------------------------------------------------------------- data construction
def test_splice_ratio(R):
    # RBI Handbook: 2022-23 real GDP Rs 161.6 vs 261.2 lakh crore under the two bases
    assert close(R["splice"]["k_real"], 1.6157, 4)


def test_phase_averages(R):  # Table 1
    ph = R["phases"]
    assert close(ph["Growth"]["Pandemic (2020Q2–2021Q1)"], -6.83, 2)
    assert close(ph["CPI infl."]["2025 easing (2025Q1–2025Q4)"], 2.23, 2)
    assert close(ph["Repo (avg)"]["Hold (2023Q2–2024Q4)"], 6.50, 2)


# ---------------------------------------------------------------- Q1
def test_output_gap_episodes(R):  # Table 2
    g = R["gaps_episode"]
    assert close(g["HP"]["2025 avg"], 0.74, 2)
    assert close(g["Hamilton"]["2025 avg"], 4.56, 2)
    assert close(g["Hamilton (ex-COVID est.)"]["2025 avg"], 2.80, 2)
    assert close(g["Trend+break"]["2020Q2 (trough)"], -20.91, 2)
    assert close(g["HP real-time"]["FY2022-23 avg"], 2.29, 2)
    assert close(R["trend_break"]["level_break"], -14.8, 1)


def test_inflation_equation(R):  # Table 3
    hp = R["inflation"]["HP"]
    assert close(hp["rho"], 0.656, 3)
    assert close(hp["comb"]["est"], -0.066, 3)
    assert close(hp["pF"], 0.010, 3)
    assert close(R["inflation"]["HPexCOVID"]["comb"]["est"], -0.282, 3)


# ---------------------------------------------------------------- Q2
def test_taylor_rule(R):  # Table 4
    t = R["taylor"]
    assert close(t["i^A"]["Tightening 2022Q2–2023Q1"], 8.50, 2)
    assert close(t["i−i^A"]["Easing 2025Q1–2025Q4"], 2.58, 2)
    assert close(t["i^B"]["Tightening 2022Q2–2023Q1"], 12.64, 2)


def test_reaction_function(R):  # Table 6
    b = R["reaction"]["base"]
    assert close(b["rho"], 0.921, 3)
    assert close(b["LRpi"]["est"], 0.98, 2) and close(b["LRpi"]["se"], 0.39, 2)
    assert close(b["LRy"]["est"], 0.39, 2)
    assert close(R["reaction"]["exCOVID"]["LRpi"]["est"], 0.61, 2)
    assert close(R["reaction_sim2025"]["2025Q4"]["sim"], 5.62, 2)


# ---------------------------------------------------------------- Q3
@pytest.mark.parametrize("rate,cum,tight,ease", [
    ("walr_fresh", 0.82, 0.71, 0.96), ("walr_out", 0.56, 0.60, 0.51),
    ("mclr_1y", 0.68, 0.44, 1.02), ("wadtdr_out", 0.85, 0.70, 1.06)])
def test_distributed_lag(R, rate, cum, tight, ease):  # Table 8
    d = R["dist_lag"][rate]
    assert close(d["cum"]["est"], cum, 2)
    assert close(d["cum_tight"], tight, 2) and close(d["cum_ease"], ease, 2)


def test_passthrough_windows(R):  # Table 7
    p = R["passthrough"]
    assert close(p["PT (WALR-F)"]["T1: Mar-22→Mar-23"], 0.68, 2)
    assert close(p["PT (WADTDR-O)"]["T2: Mar-22→Mar-24"], 0.77, 2)
    assert close(p["PT (WALR-F)"]["E3: Feb-25→Aug-26‡"], 0.63, 2)


# ---------------------------------------------------------------- Q4
def test_fiscal_impulse(R):  # Table 10
    f = R["fiscal"]
    assert close(f["GFD_y"][2021], 9.16, 2)
    assert close(f["FI"][2021], 3.01, 2)
    assert close(f["FI"][2022], -1.89, 2)
    assert close(f["FI"][2026], -0.52, 2)


def test_expenditure_growth(R):  # Table 11
    e = R["expenditure"]
    assert close(e["b"]["bK"], 3.075, 3) and close(e["b"]["bR"], 0.088, 3)
    assert close(e["c"]["cumK"]["est"], 0.86, 2)


# ---------------------------------------------------------------- Q5
def test_consumption(R):  # Tables 12-13
    c = R["consumption"]
    assert close(c["qoq"]["comb"]["est"], 0.73, 2)
    assert close(c["mpc"]["comb"]["est"], 0.48, 2)


def test_tax(R):  # Table 14
    t = R["tax"]
    assert close(t["m5"]["TAXCUT"]["b"], -0.862, 3)
    assert close(t["m2"]["cum_pit"]["est"], -1.12, 2)
