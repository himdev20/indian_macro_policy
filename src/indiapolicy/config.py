"""Paths, episode definitions and calibrations shared by every question."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
OUT = ROOT / "output"
FIG = OUT / "figures"
TAB = OUT / "tables"

# Base-year break in the quarterly national accounts (MoSPI 2022-23 base).
BASE_BREAK = "2022-04-01"
# Pandemic quarters treated as outliers in several robustness checks.
COVID_START, COVID_END = "2020-04-01", "2021-04-01"

# Episode windows (quarter-start dates, inclusive).
PHASES = {
    "Pre-pandemic (2018Q2–2020Q1)": ("2018-04-01", "2020-01-01"),
    "Pandemic (2020Q2–2021Q1)": ("2020-04-01", "2021-01-01"),
    "Recovery (2021Q2–2022Q1)": ("2021-04-01", "2022-01-01"),
    "Tightening (2022Q2–2023Q1)": ("2022-04-01", "2023-01-01"),
    "Hold (2023Q2–2024Q4)": ("2023-04-01", "2024-10-01"),
    "2025 easing (2025Q1–2025Q4)": ("2025-01-01", "2025-10-01"),
    "2026 (2026Q1–2026Q3)": ("2026-01-01", "2026-07-01"),
}
POLICY_EPISODES = {
    "Pre-tightening 2021Q4–2022Q1": ("2021-10-01", "2022-01-01"),
    "Tightening 2022Q2–2023Q1": ("2022-04-01", "2023-01-01"),
    "Hold 2023Q2–2024Q4": ("2023-04-01", "2024-10-01"),
    "Easing 2025Q1–2025Q4": ("2025-01-01", "2025-10-01"),
    "2026Q1–2026Q2": ("2026-01-01", "2026-04-01"),
    "2026Q3†": ("2026-07-01", "2026-07-01"),
}
SHADED = [("2020-03-01", "2021-06-30"), ("2022-05-01", "2023-02-28"), ("2025-02-01", "2025-12-31")]

# Taylor rule: i = r* + pi + theta_pi (pi - pi*) + theta_y * gap
PI_STAR = 4.0
TAYLOR_RULES = {
    "A": dict(rstar=1.5, tp=0.5, ty=0.5, gap="HP"),          # Taylor (1993) weights, RBI r* range
    "B": dict(rstar=1.0, tp=1.0, ty=0.5, gap="Hamilton"),    # hawkish
    "C": dict(rstar=1.9, tp=0.5, ty=1.0, gap="Trend+break"), # higher r*, more output weight
}

# Fiscal: semi-elasticity of the Centre's primary balance to the output gap.
FISCAL_EPS = 0.10
FISCAL_EPS_ALT = 0.20
NOMINAL_GROWTH_BE_2027 = 0.10  # Union Budget 2026-27 assumption

# Inference: Newey-West lags
HAC_Q, HAC_A = 4, 1

# Plot palette (validated categorical slots; ink for the actual series)
INK = "#1f1f1e"
C1, C2, C3, C4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
SHADE = "#eda100"
