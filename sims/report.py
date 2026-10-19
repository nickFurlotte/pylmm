"""Loading results and writing the committed summaries (needs pandas).

Output layout for study <name>::

    sims/results/<name>/raw/          JSONL raw results + run log (gitignored)
    sims/results/<name>/summary.csv   committed: one row per cell (full run)
    sims/results/<name>/summary.md    committed: same table + notes
    sims/results/<name>/*.png         committed: a few figures
    sims/results/<name>/quick/        report of a --quick run (gitignored)
"""

from __future__ import annotations

import json
import os
from typing import Iterable, List, Optional, Sequence

import numpy as np

from sims import runner


def load_results(name: str, quick: bool = False, mod=None):
    """Finished replicates of a study as a DataFrame.

    With ``mod`` (the study module; ``report`` always passes it) only rows of
    the current plan and STUDY_VERSION are kept (``runner.current_rows``), so
    rows left over from an older grid or version never leak into a summary.
    Without it every row of ``raw/<mode>.jsonl`` is returned.

    Columns: cell_id, rep, seed, elapsed, one column per param, one per
    metric.  List-valued metrics stay as Python lists (object columns);
    list/dict-valued *params* become their canonical JSON string so they can
    be grouped on.
    """
    import pandas as pd

    if mod is not None:
        rows = runner.current_rows(name, quick, mod=mod)
    else:
        rows = runner.read_rows(runner.raw_path(name, quick))
    recs = []
    for r in rows:
        rec = {
            "cell_id": r["cell_id"],
            "rep": r["rep"],
            "seed": r["seed"],
            "elapsed": r.get("elapsed", np.nan),
        }
        for k, v in r["params"].items():
            rec[k] = runner.canonical(v) if isinstance(v, (list, dict)) else v
        rec.update(r["metrics"])
        recs.append(rec)
    return pd.DataFrame(recs)


def param_columns(df, mod=None) -> List[str]:
    """Names of the param columns (from the study grid if given)."""
    if mod is not None:
        keys = []
        for g in runner.study_grid(mod, False) + runner.study_grid(mod, True):
            keys += [k for k in g if k not in keys]
        return [k for k in keys if k in df.columns]
    raise ValueError("pass the study module")


def scalar_metric_columns(df, exclude: Sequence[str] = ()) -> List[str]:
    skip = {"cell_id", "rep", "seed", "elapsed"} | set(exclude)
    out = []
    for c in df.columns:
        if c in skip:
            continue
        if df[c].dtype.kind in "fiub":
            out.append(c)
    return out


def summarize(df, by: Sequence[str], metrics: Optional[Sequence[str]] = None, se: bool = True):
    """Per-cell mean (and standard error over replicates) of scalar metrics.

    Returns one row per combination of ``by`` with columns ``n_reps``,
    ``<metric>`` (mean) and ``<metric>_se``.
    """
    import pandas as pd

    by = list(by)
    metrics = (
        list(metrics)
        if metrics is not None
        else [c for c in scalar_metric_columns(df) if c not in by]
    )
    g = df.groupby(by, sort=False, dropna=False)
    out = g[metrics].mean()
    if se:
        sem = g[metrics].sem().add_suffix("_se")
        out = pd.concat([out, sem], axis=1)
        out = out[[c for m in metrics for c in (m, m + "_se")]]
    out.insert(0, "n_reps", g.size())
    return out.reset_index()


def markdown_table(df, floatfmt: str = "%.4g") -> str:
    """Minimal GitHub markdown table (no tabulate dependency)."""
    cols = list(df.columns)

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "" if np.isnan(v) else floatfmt % v
        return str(v)

    lines = [
        "| " + " | ".join(str(c) for c in cols) + " |",
        "|" + "|".join("---" for _ in cols) + "|",
    ]
    for row in df.itertuples(index=False):
        lines.append("| " + " | ".join(fmt(v) for v in row) + " |")
    return "\n".join(lines) + "\n"


def output_dir(name: str, quick: bool) -> str:
    d = runner.study_dir(name)
    if quick:
        d = os.path.join(d, "quick")
    os.makedirs(d, exist_ok=True)
    return d


def write_summary(
    summary,
    outdir: str,
    title: str,
    notes: str = "",
    extra_tables: Iterable = (),
    md_columns: Optional[Sequence[str]] = None,
) -> None:
    """Write summary.csv (all columns) and summary.md (title, run info, notes,
    then the table restricted to ``md_columns`` if given, plus any
    ``extra_tables`` given as (title, DataFrame) pairs)."""
    summary.to_csv(os.path.join(outdir, "summary.csv"), index=False)
    runs = os.path.join(
        outdir if os.path.basename(outdir) != "quick" else os.path.dirname(outdir),
        "raw",
        "runs.jsonl",
    )
    info = ""
    if os.path.exists(runs):
        last = [json.loads(line) for line in open(runs) if line.strip()]
        mode = "quick" if os.path.basename(outdir) == "quick" else "full"
        last = [r for r in last if r.get("mode") == mode]
        if last:
            r = last[-1]
            total = sum(x["task_cpu_s"] for x in last)
            info = (
                "_Last run: %s, git %s, jobs=%d, wall %.0fs; total task time "
                "over all runs %.0fs._\n\n" % (r["time"], r["git"], r["jobs"], r["wall_s"], total)
            )
    parts = ["# %s\n\n" % title, info]
    if notes:
        parts.append(notes.strip() + "\n\n")
    parts.append(markdown_table(summary if md_columns is None else summary[list(md_columns)]))
    for t_title, t in extra_tables:
        parts.append("\n## %s\n\n" % t_title)
        parts.append(markdown_table(t))
    with open(os.path.join(outdir, "summary.md"), "w") as f:
        f.write("".join(parts))


def default_report(df, outdir: str, mod=None, title: str = "") -> None:
    """Mean/SE of every scalar metric per cell -> summary.csv / summary.md."""
    by = param_columns(df, mod) if mod is not None else []
    s = summarize(df, by) if by else df.describe().T.reset_index()
    write_summary(s, outdir, title or "Summary")


def report(ref: str, quick: bool = False) -> str:
    """CLI entry: load results, call the study's ``report(df, outdir)``."""
    mod = runner.load_study(ref)
    name = runner.study_name(ref, mod)
    df = load_results(ref, quick, mod=mod)
    if df.empty:
        raise SystemExit(
            "no results for %s (%s); run it first" % (name, "quick" if quick else "full")
        )
    outdir = output_dir(name, quick)
    if hasattr(mod, "report"):
        mod.report(df, outdir)
    else:
        default_report(df, outdir, mod, title=name)
    return outdir
