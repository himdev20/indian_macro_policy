"""Write tables as LaTeX (booktabs, for the report) and Markdown (for GitHub)."""
import numpy as np
import pandas as pd

from .config import TAB


def _star(p):
    return "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else ""


def _fmt(x, fmt):
    if isinstance(x, (float, np.floating)):
        return "–" if np.isnan(x) else fmt.format(x)
    return str(x)


def _write(df, name, title, note=""):
    """Write a string DataFrame to output/tables/{name}.md and .tex."""
    TAB.mkdir(parents=True, exist_ok=True)
    cols = [df.index.name or ""] + list(df.columns)
    md = [f"**{title}**", ""] + ([note, ""] if note else []) + ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for i, r in df.iterrows():
        md.append("| " + " | ".join([str(i)] + list(r.values)) + " |")
    (TAB / f"{name}.md").write_text("\n".join(md) + "\n")
    esc = lambda s: str(s).replace("%", r"\%").replace("&", r"\&")
    tex = ["\\begin{tabular}{l" + "c" * df.shape[1] + "}", "\\toprule",
           " & ".join(esc(c) for c in cols) + r" \\", "\\midrule"]
    for i, r in df.iterrows():
        if i == "__rule__":
            tex.append("\\midrule"); continue
        tex.append(" & ".join(esc(c) for c in [i] + list(r.values)) + r" \\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    (TAB / f"{name}.tex").write_text("\n".join(tex) + "\n")


def frame_table(df, name, title, fmt="{:.2f}"):
    """Numeric DataFrame -> table (also saved as CSV)."""
    TAB.mkdir(parents=True, exist_ok=True)
    df.to_csv(TAB / f"{name}.csv")
    s = df.apply(lambda col: col.map(lambda x: _fmt(x, fmt)))
    s.index.name = df.index.name
    _write(s, name, title)


def reg_table(models, names, labels, name, title, extra=None):
    """Regression table: coefficient with significance stars, (SE) underneath.

    labels: list of (param_name, display_label); extra: list of (label, [values]).
    """
    rows, idx = [], []
    for v, lab in labels:
        if not any(v in m.params.index for m in models):
            continue
        rows.append([f"{m.params[v]:.3f}{_star(m.pvalues[v])}" if v in m.params.index else "" for m in models])
        rows.append([f"({m.bse[v]:.3f})" if v in m.params.index else "" for m in models])
        idx += [lab, ""]
    rows.append([str(int(m.nobs)) for m in models]); idx.append("Observations")
    rows.append([f"{m.rsquared_adj:.3f}" for m in models]); idx.append("R² (adj.)")
    rows.append([f"{m.aic:.1f}" for m in models]); idx.append("AIC")
    for lab, vals in (extra or []):
        rows.append(list(vals)); idx.append(lab)
    # make blank SE-row labels unique for DataFrame indexing
    idx = [lab if lab else " " * (k + 1) for k, lab in enumerate(idx)]
    df = pd.DataFrame(rows, index=idx, columns=names)
    _write(df, name, title, note="<sub>\\*\\*\\* p<0.01, \\*\\* p<0.05, \\* p<0.1. Standard errors in parentheses.</sub>")
