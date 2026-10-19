"""Grid runner: param grid x replicates -> append-only JSONL, resumable, parallel.

A *study* is a module (``sims.studies.<name>``, or a path to a .py file)
defining::

    GRID        dict of lists (cartesian product) or list of param dicts
    QUICK_GRID  same, tiny, for --quick smoke runs
    REPS, QUICK_REPS                    replicates per cell
    def run_cell(params: dict, seed: int) -> dict   # scalar metrics (+ small lists)
    def report(df, outdir) -> None      # optional, see sims.report
    SEED_PARAMS  optional list of param names that alone determine the seed
                 (common random numbers across cells that differ only in the
                 other params, e.g. the method)
    STUDY_VERSION  optional int (default 1).  Bump it whenever run_cell's
                 output changes meaning for the same params (a new metric
                 definition, a bug fix in the simulation).  It is stored in
                 every row, and from version 2 on it is part of cell_id, so a
                 resumed run never mixes rows of two versions and the loaders
                 (``current_rows``, ``report.load_results``) ignore stale rows.

Instead of the grid constants a study may define ``grid(quick: bool)``
returning a list of param dicts, and ``reps(quick: bool)``.

Seeds: ``task_seed(study, params, rep)`` hashes (study name, canonical JSON
of the seed params, rep) with SHA-256, so a given replicate of a given cell
always gets the same seed regardless of grid order, --jobs or resumption.

Storage: ``sims/results/<study>/raw/<mode>.jsonl`` (mode = quick | full), one
JSON object per finished replicate::

    {"cell_id", "rep", "seed", "study_version", "params": {...}, "metrics": {...},
     "elapsed", "time"}

(rows written before STUDY_VERSION existed have no ``study_version``; they
count as version 1).  Rows are appended (and flushed) as soon as a chunk
finishes; on restart, (cell_id, rep) pairs already present are skipped, so
rerunning a study in place computes only the missing replicates.  Failed
replicates are never written to ``<mode>.jsonl``: they go to
``raw/<mode>.errors.jsonl`` with the traceback, and since they stay missing
they are retried on the next run (the errors file is rewritten with that
run's failures only).  ``--cells ID [ID ...]`` restricts a run to some cells
(ids as printed by ``python -m sims plan``).  ``raw/runs.jsonl`` logs each invocation (git SHA, args, wall time).
Set ``SIMS_RESULTS_DIR`` to put results elsewhere (the tests do).
"""

from __future__ import annotations

import concurrent.futures as cf
import datetime as _dt
import hashlib
import importlib
import importlib.util
import itertools
import json
import math
import multiprocessing
import os
import subprocess
import sys
import time
import traceback
from collections.abc import Iterable
from typing import Any

import numpy as np

import sims  # noqa: F401  (pins BLAS threads before numpy loads)

SIMS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(SIMS_DIR)
MAX_JOBS = 6  # 8-core laptop (4 performance cores), 16 GB RAM


# ---------------------------------------------------------------------------
# studies
# ---------------------------------------------------------------------------

_LOADED: dict[str, Any] = {}


def load_study(ref: str):
    """Import a study by name (``example_null``), dotted module path, or
    path to a .py file."""
    if ref in _LOADED:
        return _LOADED[ref]
    if ref.endswith(".py") or os.sep in ref:
        path = os.path.abspath(ref)
        modname = "sims_study_" + os.path.splitext(os.path.basename(path))[0]
        spec = importlib.util.spec_from_file_location(modname, path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules[modname] = mod
        spec.loader.exec_module(mod)
    elif "." in ref:
        mod = importlib.import_module(ref)
    else:
        mod = importlib.import_module("sims.studies." + ref)
    if not hasattr(mod, "run_cell"):
        raise AttributeError("study %r has no run_cell(params, seed)" % ref)
    _LOADED[ref] = mod
    return mod


def study_name(ref: str, mod=None) -> str:
    mod = mod or load_study(ref)
    return getattr(mod, "NAME", None) or os.path.splitext(os.path.basename(ref))[0].split(".")[-1]


def expand_grid(grid) -> list[dict]:
    """dict of lists -> cartesian product (in key order); list -> as is."""
    if isinstance(grid, dict):
        keys = list(grid)
        vals = [v if isinstance(v, (list, tuple)) else [v] for v in grid.values()]
        return [dict(zip(keys, combo)) for combo in itertools.product(*vals)]
    return [dict(g) for g in grid]


def study_grid(mod, quick: bool) -> list[dict]:
    if hasattr(mod, "grid"):
        return expand_grid(mod.grid(quick))
    return expand_grid(mod.QUICK_GRID if quick else mod.GRID)


def study_reps(mod, quick: bool) -> int:
    if hasattr(mod, "reps"):
        return int(mod.reps(quick))
    return int(mod.QUICK_REPS if quick else mod.REPS)


# ---------------------------------------------------------------------------
# ids and seeds
# ---------------------------------------------------------------------------


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError("not JSON serialisable: %r (%s)" % (o, type(o).__name__))


def canonical(obj) -> str:
    return json.dumps(obj, sort_keys=True, default=_json_default, separators=(",", ":"))


def cell_id(params: dict, version: int = 1) -> str:
    """12-hex id of a cell.  Version 1 hashes the params alone (the ids of
    every raw file written before STUDY_VERSION existed); later versions hash
    the version too, so their rows never collide with older ones."""
    key = (
        canonical(params)
        if int(version) == 1
        else canonical({"params": params, "study_version": int(version)})
    )
    return hashlib.sha1(key.encode()).hexdigest()[:12]


def study_version(mod) -> int:
    return int(getattr(mod, "STUDY_VERSION", 1))


def task_seed(study: str, params: dict, rep: int, seed_params=None) -> int:
    """Deterministic 63-bit seed for one replicate of one cell."""
    if seed_params is not None:
        params = {k: params[k] for k in seed_params if k in params}
    h = hashlib.sha256(("%s|%s|%d" % (study, canonical(params), rep)).encode()).digest()
    return int.from_bytes(h[:8], "little") >> 1


# ---------------------------------------------------------------------------
# storage
# ---------------------------------------------------------------------------


def results_root() -> str:
    return os.environ.get("SIMS_RESULTS_DIR") or os.path.join(SIMS_DIR, "results")


def study_dir(name: str) -> str:
    return os.path.join(results_root(), name)


def raw_path(name: str, quick: bool) -> str:
    return os.path.join(study_dir(name), "raw", ("quick" if quick else "full") + ".jsonl")


def read_rows(path: str) -> list[dict]:
    rows = []
    if not os.path.exists(path):
        return rows
    with open(path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                sys.stderr.write(
                    "warning: skipping unreadable line %d of %s "
                    "(interrupted write?)\n" % (i + 1, path)
                )
    return rows


def _append(path: str, rows: Iterable[dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r, default=_json_default) + "\n")
        f.flush()
        os.fsync(f.fileno())


def row_version(row: dict) -> int:
    return int(row.get("study_version", 1))


def current_rows(ref: str, quick: bool = False, mod=None) -> list[dict]:
    """Rows of ``raw/<mode>.jsonl`` that belong to the study as it is now:
    cell_id in the current grid and row version == STUDY_VERSION (extra
    replicates from a ``--reps`` run are kept).  Rows of a cell that is no
    longer in the grid, of an older version, or duplicates of a (cell_id,
    rep) are dropped with a note on stderr, so summaries never mix designs."""
    mod = mod or load_study(ref)
    name = study_name(ref, mod)
    ver = study_version(mod)
    want = {cell_id(p, ver) for p in study_grid(mod, quick)}
    rows = read_rows(raw_path(name, quick))
    keep, seen = [], set()
    for r in rows:
        k = (r["cell_id"], r["rep"])
        if r["cell_id"] in want and row_version(r) == ver and k not in seen:
            seen.add(k)
            keep.append(r)
    if len(keep) < len(rows):
        sys.stderr.write(
            "note: %s: ignoring %d of %d raw rows not in the current plan "
            "(grid or STUDY_VERSION %d) or duplicated\n"
            % (name, len(rows) - len(keep), len(rows), ver)
        )
    return keep


def git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"],
            cwd=REPO,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip()
        return sha + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


# ---------------------------------------------------------------------------
# execution
# ---------------------------------------------------------------------------


def _execute(ref: str, tasks: list[tuple]) -> list[dict]:
    """Run tasks [(cell_id, params, rep, seed), ...] in this process."""
    mod = load_study(ref)
    ver = study_version(mod)
    out = []
    for cid, params, rep, seed in tasks:
        t0 = time.perf_counter()
        try:
            metrics = mod.run_cell(dict(params), int(seed))
            if not isinstance(metrics, dict):
                raise TypeError("run_cell must return a dict, got %s" % type(metrics).__name__)
            clash = set(metrics) & set(params)
            if clash:
                raise KeyError("metric names clash with param names: %s" % sorted(clash))
            # round-trip now so a non-serialisable metric fails here, not later
            metrics = json.loads(json.dumps(metrics, default=_json_default))
            out.append(
                {
                    "ok": True,
                    "cell_id": cid,
                    "rep": rep,
                    "seed": seed,
                    "study_version": ver,
                    "params": params,
                    "metrics": metrics,
                    "elapsed": time.perf_counter() - t0,
                    "time": _dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
        except Exception:
            out.append(
                {
                    "ok": False,
                    "cell_id": cid,
                    "rep": rep,
                    "seed": seed,
                    "study_version": ver,
                    "params": params,
                    "error": traceback.format_exc(),
                    "elapsed": time.perf_counter() - t0,
                    "time": _dt.datetime.now().isoformat(timespec="seconds"),
                }
            )
    return out


def _worker_init():
    import sims as _s

    _s.pin_blas_threads()


def plan(ref: str, quick: bool, reps: int | None = None, mod=None) -> list[tuple]:
    """All (cell_id, params, rep, seed) tasks of the study, in grid order."""
    mod = mod or load_study(ref)
    name = study_name(ref, mod)
    seed_params = getattr(mod, "SEED_PARAMS", None)
    n_reps = study_reps(mod, quick) if reps is None else reps
    ver = study_version(mod)
    tasks = []
    for params in study_grid(mod, quick):
        cid = cell_id(params, ver)
        for rep in range(n_reps):
            tasks.append((cid, params, rep, task_seed(name, params, rep, seed_params)))
    return tasks


def run(
    ref: str,
    quick: bool = False,
    jobs: int = 1,
    reps: int | None = None,
    force: bool = False,
    chunk: int | None = None,
    verbose: bool = True,
    job_cap: int = MAX_JOBS,
    cells: Iterable[str] | None = None,
) -> dict:
    """Run (or resume) a study.  Returns a small summary dict.

    Only replicates missing from ``raw/<mode>.jsonl`` run (earlier failures
    included).  ``cells``: restrict the run to these cell ids."""
    mod = load_study(ref)
    name = study_name(ref, mod)
    path = raw_path(name, quick)
    err_path = path.replace(".jsonl", ".errors.jsonl")
    if jobs > job_cap:
        sys.stderr.write(
            "note: --jobs %d capped at %d (laptop budget; pass a "
            "higher job_cap/--no-job-cap to override)\n" % (jobs, job_cap)
        )
        jobs = job_cap
    jobs = max(1, int(jobs))
    if force:
        for p in (path, err_path):
            if os.path.exists(p):
                os.remove(p)

    tasks = plan(ref, quick, reps, mod)
    if cells is not None:
        cells = set(cells)
        unknown = cells - {t[0] for t in tasks}
        if unknown:
            raise ValueError(
                "unknown cell ids for %s: %s (see `python -m sims plan`)"
                % (name, ", ".join(sorted(unknown)))
            )
        tasks = [t for t in tasks if t[0] in cells]
    done = {(r["cell_id"], r["rep"]) for r in read_rows(path)}
    todo = [t for t in tasks if (t[0], t[2]) not in done]
    if os.path.exists(err_path):
        os.remove(err_path)  # failures are retried; keep only this run's
    log = lambda msg: verbose and sys.stderr.write(msg + "\n")  # noqa: E731
    log(
        "%s [%s]: %d cells x %d reps = %d tasks; %d already done, %d to run, jobs=%d"
        % (
            name,
            "quick" if quick else "full",
            len({t[0] for t in tasks}),
            study_reps(mod, quick) if reps is None else reps,
            len(tasks),
            len(tasks) - len(todo),
            len(todo),
            jobs,
        )
    )

    # chunks: consecutive reps of one cell, so a worker reuses cached data
    by_cell: dict[str, list[tuple]] = {}
    for t in todo:
        by_cell.setdefault(t[0], []).append(t)
    chunks = []
    for cell_tasks in by_cell.values():
        size = chunk or (1 if jobs == 1 else max(1, min(10, math.ceil(len(cell_tasks) / jobs))))
        for i in range(0, len(cell_tasks), size):
            chunks.append(cell_tasks[i : i + size])

    t0 = time.time()
    n_ok = n_err = 0
    task_time = 0.0

    def handle(results):
        nonlocal n_ok, n_err, task_time
        good = [{k: v for k, v in r.items() if k != "ok"} for r in results if r["ok"]]
        bad = [{k: v for k, v in r.items() if k != "ok"} for r in results if not r["ok"]]
        if good:
            _append(path, good)
        if bad:
            _append(err_path, bad)
            for b in bad:
                log(
                    "ERROR cell %s rep %d: %s"
                    % (b["cell_id"], b["rep"], b["error"].strip().splitlines()[-1])
                )
        n_ok += len(good)
        n_err += len(bad)
        task_time += sum(r["elapsed"] for r in results)
        k = n_ok + n_err
        el = time.time() - t0
        eta = el / k * (len(todo) - k) if k else 0
        log("  %d/%d done (%d errors), %.0fs elapsed, ~%.0fs left" % (k, len(todo), n_err, el, eta))

    if jobs == 1 or len(chunks) <= 1:
        for c in chunks:
            handle(_execute(ref, c))
    else:
        ctx = multiprocessing.get_context("spawn")
        with cf.ProcessPoolExecutor(
            max_workers=jobs, mp_context=ctx, initializer=_worker_init
        ) as ex:
            futs = [ex.submit(_execute, ref, c) for c in chunks]
            for f in cf.as_completed(futs):
                handle(f.result())

    wall = time.time() - t0
    summary = {
        "study": name,
        "mode": "quick" if quick else "full",
        "tasks": len(tasks),
        "study_version": study_version(mod),
        "cells": None if cells is None else sorted(cells),
        "skipped": len(tasks) - len(todo),
        "ran": n_ok,
        "errors": n_err,
        "jobs": jobs,
        "wall_s": round(wall, 2),
        "task_cpu_s": round(task_time, 2),
        "git": git_sha(),
        "time": _dt.datetime.now().isoformat(timespec="seconds"),
        "path": os.path.relpath(path, REPO) if path.startswith(REPO) else path,
    }
    _append(os.path.join(os.path.dirname(path), "runs.jsonl"), [summary])
    log(
        "%s: ran %d, errors %d, wall %.1fs (sum of task times %.1fs) -> %s"
        % (name, n_ok, n_err, wall, task_time, summary["path"])
    )
    return summary
