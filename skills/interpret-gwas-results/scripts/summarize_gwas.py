"""Summarise a `pylmm gwas` output file: counts, lambda_GC, top hits, optional QQ/Manhattan plots.

    python skills/interpret-gwas-results/scripts/summarize_gwas.py results.tsv \
        --bfile geno --top 10 --plot results.png

``--bfile PREFIX`` joins the chromosome, position and alleles of ``PREFIX.bim``
(the output has one line per ``.bim`` line, in order).  ``--plot`` needs
matplotlib.  Only numpy and scipy otherwise; importable as a module
(`read_results`, `lambda_gc`, `summarize`).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import stats

COLUMNS = ("SNP_ID", "BETA", "BETA_SD", "F_STAT", "P_VALUE")
CHI2_MEDIAN = float(stats.chi2.ppf(0.5, 1))  # 0.4549...


def read_results(path: str) -> dict[str, np.ndarray]:
    """The columns of a `pylmm gwas` output file (``nan`` for untestable SNPs)."""
    with open(path) as f:
        header = f.readline().rstrip("\n").split("\t")
        if tuple(header) != COLUMNS:
            raise ValueError("%s: not a pylmm gwas output (header %r)" % (path, header))
        rows = [line.rstrip("\n").split("\t") for line in f if line.strip()]
    ids = np.array([r[0] for r in rows], dtype=str)
    vals = np.array([[float(x) for x in r[1:]] for r in rows], dtype=float).reshape(-1, 4)
    out = {"SNP_ID": ids}
    for j, name in enumerate(COLUMNS[1:]):
        out[name] = vals[:, j]
    return out


def lambda_gc(p: np.ndarray) -> float:
    """Genomic control: median of the 1-df chi^2 of the p-values / 0.4549 (nan ignored)."""
    p = np.asarray(p, dtype=float)
    p = p[~np.isnan(p)]
    if not len(p):
        return float("nan")
    return float(np.median(stats.chi2.isf(p, 1)) / CHI2_MEDIAN)


def read_bim(prefix: str) -> dict[str, np.ndarray]:
    rows = [line.split()[:6] for line in open(prefix + ".bim") if line.strip()]
    a = np.array(rows, dtype=str).reshape(-1, 6)
    return {"chrom": a[:, 0], "pos": a[:, 3].astype(np.int64), "a1": a[:, 4], "a2": a[:, 5]}


def summarize(res: dict[str, np.ndarray], bim: dict | None = None, top: int = 10) -> dict:
    p = res["P_VALUE"]
    ok = ~np.isnan(p)
    m = int(ok.sum())
    order = np.argsort(np.where(ok, p, np.inf))[: min(top, m)]
    hits = []
    for j in order:
        h = {
            "snp": str(res["SNP_ID"][j]),
            "p": float(p[j]),
            "t": float(res["F_STAT"][j]),
            "beta_std": float(res["BETA"][j]),
        }
        if bim is not None:
            h.update(
                chrom=str(bim["chrom"][j]), pos=int(bim["pos"][j]), effect_allele=str(bim["a2"][j])
            )
        hits.append(h)
    return {
        "n_snps": len(p),
        "n_tested": m,
        "n_nan": int((~ok).sum()),
        "lambda_gc": lambda_gc(p),
        "bonferroni_0.05": 0.05 / m if m else float("nan"),
        "n_below_bonferroni": int((p[ok] < 0.05 / m).sum()) if m else 0,
        "min_p": float(p[ok].min()) if m else float("nan"),
        "top": hits,
    }


def plot(res: dict[str, np.ndarray], bim: dict | None, path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = res["P_VALUE"]
    ok = ~np.isnan(p)
    lp = -np.log10(np.clip(p[ok], 1e-300, 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={"width_ratios": [3, 1]})
    if bim is not None:
        chrom = bim["chrom"][ok]
        _, first = np.unique(chrom, return_index=True)
        colors = np.zeros(ok.sum())
        for k, c in enumerate(chrom[np.sort(first)]):
            colors[chrom == c] = k % 2
        ax1.scatter(np.arange(ok.sum()), lp, c=colors, s=3, cmap="coolwarm")
        ax1.set_xlabel("SNP (in .bim order; colours alternate by chromosome)")
    else:
        ax1.scatter(np.arange(ok.sum()), lp, s=3)
        ax1.set_xlabel("SNP (file order)")
    ax1.axhline(-np.log10(0.05 / ok.sum()), color="grey", ls="--", lw=1)
    ax1.set_ylabel("-log10 p")
    exp = -np.log10((np.arange(1, len(lp) + 1) - 0.5) / len(lp))
    ax2.scatter(exp, np.sort(lp)[::-1], s=3)
    ax2.plot([0, exp.max()], [0, exp.max()], color="grey", lw=1)
    ax2.set_xlabel("expected -log10 p")
    ax2.set_title("QQ, lambda_GC = %.3f" % lambda_gc(p))
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("results", help="output file of pylmm gwas")
    ap.add_argument(
        "--bfile", help="genotype prefix: join chromosome/position/alleles of PREFIX.bim"
    )
    ap.add_argument("--top", type=int, default=10, help="number of top SNPs to list (default 10)")
    ap.add_argument("--plot", help="write a Manhattan + QQ PNG here (needs matplotlib)")
    args = ap.parse_args(argv)
    res = read_results(args.results)
    bim = read_bim(args.bfile) if args.bfile else None
    if bim is not None and len(bim["pos"]) != len(res["P_VALUE"]):
        sys.exit(
            "%s.bim has %d SNPs, the results %d"
            % (args.bfile, len(bim["pos"]), len(res["P_VALUE"]))
        )
    s = summarize(res, bim, args.top)
    print("SNPs: %d (%d tested, %d nan)" % (s["n_snps"], s["n_tested"], s["n_nan"]))
    print("lambda_GC: %.3f" % s["lambda_gc"])
    print(
        "min p: %.3g; Bonferroni 0.05 threshold %.3g, %d SNPs below it"
        % (s["min_p"], s["bonferroni_0.05"], s["n_below_bonferroni"])
    )
    print("top SNPs (beta_std: per SD of the A2 allele; t = F_STAT):")
    for h in s["top"]:
        where = (
            " chr%s:%d A2=%s" % (h["chrom"], h["pos"], h["effect_allele"]) if "chrom" in h else ""
        )
        print(
            "  %s%s  p=%.3g  t=%.2f  beta_std=%.3g"
            % (h["snp"], where, h["p"], h["t"], h["beta_std"])
        )
    if args.plot:
        plot(res, bim, args.plot)
        print("plot: %s" % args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
