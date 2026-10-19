"""CLI: ``uv run --group sims python -m sims {run,report,list,plan} ...``."""

import sims  # noqa: F401  (pins BLAS threads first)

import argparse
import os
import sys


def _studies():
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "studies")
    return sorted(f[:-3] for f in os.listdir(d) if f.endswith(".py") and not f.startswith("_"))


def main(argv=None):
    from sims import runner

    ap = argparse.ArgumentParser(prog="python -m sims", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run or resume a study")
    r.add_argument("study", help="name in sims/studies/, dotted module or .py path")
    r.add_argument("--quick", action="store_true", help="tiny smoke-test grid")
    r.add_argument(
        "--jobs",
        "-j",
        type=int,
        default=1,
        help="worker processes (default 1, capped at %d)" % runner.MAX_JOBS,
    )
    r.add_argument(
        "--no-job-cap", action="store_true", help="allow --jobs above %d" % runner.MAX_JOBS
    )
    r.add_argument("--reps", type=int, help="override replicates per cell")
    r.add_argument(
        "--force", action="store_true", help="discard existing raw results for this mode first"
    )
    r.add_argument("--report", action="store_true", help="run the report afterwards")
    r.add_argument(
        "--cells",
        nargs="+",
        metavar="ID",
        help="run only these cell ids (from `plan`); only their missing or failed replicates run",
    )

    p = sub.add_parser("report", help="summarise results (summary.csv/md, PNGs)")
    p.add_argument("study")
    p.add_argument("--quick", action="store_true")

    pl = sub.add_parser("plan", help="print the grid and task count")
    pl.add_argument("study")
    pl.add_argument("--quick", action="store_true")

    sub.add_parser("list", help="list studies")

    a = ap.parse_args(argv)
    if a.cmd == "list":
        print("\n".join(_studies()))
        return 0
    if a.cmd == "plan":
        mod = runner.load_study(a.study)
        grid = runner.study_grid(mod, a.quick)
        ver = runner.study_version(mod)
        for g in grid:
            print(runner.cell_id(g, ver), runner.canonical(g))
        print(
            "%d cells x %d reps = %d tasks"
            % (
                len(grid),
                runner.study_reps(mod, a.quick),
                len(grid) * runner.study_reps(mod, a.quick),
            )
        )
        return 0
    if a.cmd == "run":
        s = runner.run(
            a.study,
            quick=a.quick,
            jobs=a.jobs,
            reps=a.reps,
            force=a.force,
            job_cap=10**6 if a.no_job_cap else runner.MAX_JOBS,
            cells=a.cells,
        )
        if a.report and s["ran"] + s["skipped"]:
            from sims import report

            print("report written to", report.report(a.study, quick=a.quick))
        return 1 if s["errors"] else 0
    if a.cmd == "report":
        from sims import report

        print("report written to", report.report(a.study, quick=a.quick))
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
