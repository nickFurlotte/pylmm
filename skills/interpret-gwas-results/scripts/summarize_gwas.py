"""Summarise a `pylmm gwas` output file: counts, lambda_GC, top hits or loci, optional plots.

    python skills/interpret-gwas-results/scripts/summarize_gwas.py results.tsv \
        --bfile geno --per-allele --clump-kb 1000 --top 10 --plot results.png

``--bfile PREFIX`` joins the chromosome, position and alleles of ``PREFIX.bim``
(the output has one line per ``.bim`` line, in order).

``--per-allele`` (needs ``--bfile``) converts BETA, which pylmm reports per SD
of the standardised genotype, to the effect per copy of the counted allele A2
(``.bim`` column 6), ``BETA / sd``, with ``sd`` the SD of the raw 0/1/2 A2
dosage over the called individuals of ``PREFIX.bed`` (as pylmm standardises
it); the SE likewise.  The A1 effect is the same with the sign flipped.  Not
valid for scans run with ``--remove-missing-genotypes`` (BETA is then on a
different scale for SNPs without missing calls), and approximate if the
kinship dropped individuals of the ``.fam``.

``--clump-kb KB`` groups the hits into loci by a simple distance window (no
LD): the best SNP left becomes a lead, and every SNP within KB kilobases of it
on the same chromosome joins its locus.  ``--clump-snps N`` does the same with
a window of N SNPs on either side, in file order, when there are no positions
(no ``--bfile``).  ``--clump-p`` only starts loci at SNPs below that p.

SNPs are ranked by p, ties (e.g. p = 0.0 underflows) by abs(t).  ``--plot``
draws a Manhattan plot (x = base-pair position, chromosomes side by side,
when ``--bfile`` gives positions) and a QQ plot; it needs matplotlib.  Only
numpy and scipy otherwise.  Importable as a module (`read_results`,
`lambda_gc`, `read_bim`, `dosage_sd`, `rank`, `clump`, `summarize`).
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
from scipy import stats

COLUMNS = ("SNP_ID", "BETA", "BETA_SD", "F_STAT", "P_VALUE")
CHI2_MEDIAN = float(stats.chi2.ppf(0.5, 1))  # 0.4549...
BED_MAGIC = bytes([0x6C, 0x1B, 0x01])


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
    with open(prefix + ".bim") as f:
        rows = [line.split()[:6] for line in f if line.strip()]
    a = np.array(rows, dtype=str).reshape(-1, 6)
    return {"chrom": a[:, 0], "pos": a[:, 3].astype(np.int64), "a1": a[:, 4], "a2": a[:, 5]}


def dosage_sd(prefix: str, chunk: int = 4096) -> np.ndarray:
    """SD (population, over called individuals) of each SNP's 0/1/2 A2 dosage in ``prefix.bed``."""
    with open(prefix + ".fam") as f:
        n = sum(1 for line in f if line.strip())
    with open(prefix + ".bim") as f:
        m = sum(1 for line in f if line.strip())
    nb = (n + 3) // 4
    with open(prefix + ".bed", "rb") as f:
        if f.read(3) != BED_MAGIC:
            raise ValueError("%s.bed is not a SNP-major PLINK 1 BED file" % prefix)
        data = np.frombuffer(f.read(), dtype=np.uint8)
    if data.size != nb * m:
        raise ValueError("%s.bed has %d genotype bytes, expected %d" % (prefix, data.size, nb * m))
    data = data.reshape(m, nb)
    lut = np.array([0.0, np.nan, 1.0, 2.0])  # BED 2-bit codes 00, 01 (missing), 10, 11
    sd = np.empty(m)
    for start in range(0, m, chunk):
        blk = data[start : start + chunk]
        codes = np.stack([(blk >> s) & 3 for s in (0, 2, 4, 6)], axis=2).reshape(len(blk), -1)
        g = lut[codes[:, :n]]
        called = ~np.isnan(g)
        k = called.sum(axis=1)
        mean = np.where(k > 0, np.nansum(g, axis=1) / np.maximum(k, 1), np.nan)
        var = np.nansum((g - mean[:, None]) ** 2, axis=1) / np.maximum(k, 1)
        sd[start : start + len(blk)] = np.where(k > 0, np.sqrt(var), np.nan)
    return sd


def rank(res: dict[str, np.ndarray]) -> np.ndarray:
    """Indices of the testable SNPs, best first: by p, ties (p = 0.0 underflows) by abs(t)."""
    p, t = res["P_VALUE"], res["F_STAT"]
    ok = np.flatnonzero(~np.isnan(p))
    order = np.lexsort((-np.abs(t[ok]), p[ok]))
    return ok[order]


def clump(
    res: dict[str, np.ndarray],
    bim: dict | None = None,
    kb: float | None = None,
    snps: int | None = None,
    p_max: float = 1.0,
) -> list[dict]:
    """Greedy window clumping (no LD): loci as dicts with the lead and its members.

    With ``kb`` (and ``bim``), a SNP joins the locus of the best remaining
    lead on its chromosome within ``kb`` kilobases; with ``snps``, within
    ``snps`` positions in file order (on the same chromosome if ``bim`` is
    given).  Leads are taken in `rank` order while their p < ``p_max``.
    """
    if (kb is None) == (snps is None):
        raise ValueError("give exactly one of kb (with bim) and snps")
    if kb is not None and bim is None:
        raise ValueError("kb clumping needs the .bim positions; use snps= without them")
    m = len(res["P_VALUE"])
    chrom = bim["chrom"] if bim is not None else np.zeros(m, dtype=str)
    where = bim["pos"].astype(float) / 1000.0 if kb is not None else np.arange(m, dtype=float)
    width = float(kb if kb is not None else snps)
    taken = np.zeros(m, dtype=bool)
    tested = ~np.isnan(res["P_VALUE"])
    loci = []
    for j in rank(res):
        if taken[j]:
            continue
        if res["P_VALUE"][j] >= p_max:
            break
        members = np.flatnonzero(
            (chrom == chrom[j]) & (np.abs(where - where[j]) <= width) & ~taken & tested
        )
        taken[members] = True
        loci.append({"lead": int(j), "members": members})
    return loci


def summarize(
    res: dict[str, np.ndarray],
    bim: dict | None = None,
    top: int = 10,
    sd: np.ndarray | None = None,
    loci: list[dict] | None = None,
) -> dict:
    """Counts, lambda_GC, Bonferroni, and the top SNPs (or the leads of ``loci``).

    ``sd`` (from `dosage_sd`) adds per-copy effects; ``loci`` (from `clump`)
    lists one entry per locus instead of one per SNP.
    """
    p = res["P_VALUE"]
    ok = ~np.isnan(p)
    m = int(ok.sum())
    if loci is None:
        leads = [(int(j), None) for j in rank(res)[:top]]
    else:
        leads = [(lo["lead"], lo["members"]) for lo in loci[:top]]
    hits = []
    for j, members in leads:
        h: dict = {
            "index": j,
            "snp": str(res["SNP_ID"][j]),
            "p": float(p[j]),
            "t": float(res["F_STAT"][j]),
            "beta_std": float(res["BETA"][j]),
        }
        if bim is not None:
            h.update(
                chrom=str(bim["chrom"][j]),
                pos=int(bim["pos"][j]),
                a1=str(bim["a1"][j]),
                a2=str(bim["a2"][j]),
                effect_allele=str(bim["a2"][j]),
            )
        if sd is not None:
            h.update(
                dosage_sd=float(sd[j]),
                beta_a2=float(res["BETA"][j] / sd[j]),
                se_a2=float(res["BETA_SD"][j] / sd[j]),
                beta_a1=float(-res["BETA"][j] / sd[j]),
            )
        if members is not None:
            h["n_snps"] = len(members)
            h["n_below_bonferroni"] = int((p[members] < 0.05 / m).sum()) if m else 0
            if bim is not None:
                h["start"], h["end"] = (
                    int(bim["pos"][members].min()),
                    int(bim["pos"][members].max()),
                )
            else:
                h["first"], h["last"] = int(members.min()), int(members.max())
        hits.append(h)
    return {
        "n_snps": len(p),
        "n_tested": m,
        "n_nan": int((~ok).sum()),
        "lambda_gc": lambda_gc(p),
        "bonferroni_0.05": 0.05 / m if m else float("nan"),
        "n_below_bonferroni": int((p[ok] < 0.05 / m).sum()) if m else 0,
        "min_p": float(p[ok].min()) if m else float("nan"),
        "n_loci": None if loci is None else len(loci),
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
        chrom, pos = bim["chrom"][ok], bim["pos"][ok].astype(float)
        _, first = np.unique(chrom, return_index=True)
        x = np.zeros(ok.sum())
        colors = np.zeros(ok.sum())
        ticks, labels, offset = [], [], 0.0
        for k, c in enumerate(chrom[np.sort(first)]):  # chromosomes in file order
            on = chrom == c
            x[on] = offset + pos[on] - pos[on].min()
            colors[on] = k % 2
            ticks.append(offset + (pos[on].max() - pos[on].min()) / 2)
            labels.append(str(c))
            offset += pos[on].max() - pos[on].min() + max(1.0, 0.02 * pos[on].max())
        ax1.scatter(x, lp, c=colors, s=3, cmap="coolwarm")
        ax1.set_xticks(ticks)
        ax1.set_xticklabels(labels, fontsize=7)
        ax1.set_xlabel("chromosome (x = base-pair position)")
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
    ap.add_argument("--top", type=int, default=10, help="number of SNPs/loci to list (default 10)")
    ap.add_argument(
        "--per-allele",
        action="store_true",
        help="also report BETA/SE per copy of A2 (and A1, flipped), from PREFIX.bed (needs --bfile)",
    )
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--clump-kb", type=float, help="clump hits into loci of +-KB kb (needs --bfile)")
    g.add_argument("--clump-snps", type=int, help="clump hits into loci of +-N SNPs (file order)")
    ap.add_argument(
        "--clump-p", type=float, default=1.0, help="start loci only at SNPs with p below this"
    )
    ap.add_argument("--plot", help="write a Manhattan + QQ PNG here (needs matplotlib)")
    args = ap.parse_args(argv)
    if args.per_allele and not args.bfile:
        ap.error("--per-allele needs --bfile (the dosage SDs come from PREFIX.bed)")
    if args.clump_kb is not None and not args.bfile:
        ap.error("--clump-kb needs --bfile for positions; use --clump-snps N without one")
    res = read_results(args.results)
    bim = read_bim(args.bfile) if args.bfile else None
    if bim is not None and len(bim["pos"]) != len(res["P_VALUE"]):
        sys.exit(
            "%s.bim has %d SNPs, the results %d"
            % (args.bfile, len(bim["pos"]), len(res["P_VALUE"]))
        )
    sd = dosage_sd(args.bfile) if args.per_allele else None
    loci = None
    if args.clump_kb is not None or args.clump_snps is not None:
        loci = clump(res, bim, kb=args.clump_kb, snps=args.clump_snps, p_max=args.clump_p)
    s = summarize(res, bim, args.top, sd=sd, loci=loci)
    print("SNPs: %d (%d tested, %d nan)" % (s["n_snps"], s["n_tested"], s["n_nan"]))
    print("lambda_GC: %.3f" % s["lambda_gc"])
    print(
        "min p: %.3g; Bonferroni 0.05 threshold %.3g, %d SNPs below it"
        % (s["min_p"], s["bonferroni_0.05"], s["n_below_bonferroni"])
    )
    if loci is not None:
        window = (
            "%g kb" % args.clump_kb if args.clump_kb is not None else "%d SNPs" % args.clump_snps
        )
        print("loci (+-%s windows, leads with p < %g): %d" % (window, args.clump_p, len(loci)))
    what = "loci (lead SNPs)" if loci is not None else "top SNPs"
    print("%s (beta_std: per SD of the A2 allele; t = F_STAT):" % what)
    if loci is not None and not loci:
        print("  none: no SNP has p < %g (--clump-p)" % args.clump_p)
    for h in s["top"]:
        where = (
            " chr%s:%d A1=%s A2=%s" % (h["chrom"], h["pos"], h["a1"], h["a2"])
            if "chrom" in h
            else ""
        )
        line = "  %s%s  p=%.3g  t=%.2f  beta_std=%.3g" % (
            h["snp"],
            where,
            h["p"],
            h["t"],
            h["beta_std"],
        )
        if "beta_a2" in h:
            line += "  per A2 copy %.3g (SE %.3g), per A1 copy %.3g" % (
                h["beta_a2"],
                h["se_a2"],
                h["beta_a1"],
            )
        if "n_snps" in h:
            span = (
                "%d-%d" % (h["start"], h["end"])
                if "start" in h
                else "#%d-#%d" % (h["first"], h["last"])
            )
            line += "  [%d SNPs, %s, %d below Bonferroni]" % (
                h["n_snps"],
                span,
                h["n_below_bonferroni"],
            )
        print(line)
    if args.plot:
        plot(res, bim, args.plot)
        print("plot: %s" % args.plot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
