#!/usr/bin/env python3
"""Per-family recovery score: how nearly each Glottolog family is a clade.

Why this exists
---------------
`analyze_tree_vs_glottolog.py` reports family monophyly as a yes/no count. At
3397 taxa that is brutally brittle: one misplaced language destroys a family of
642, and a tree that gets Atlantic-Congo 99.8 % right scores exactly the same
as one that shatters it. GQD has the opposite flaw -- it weights families by
their pair count, so Atlantic-Congo and Austronesian alone drive ~78 % of the
number (see that script's quartet breakdown).

This measure sits between the two. For each family it asks how close the tree
comes to recovering it, and weights every family equally:

    F1(family) = max over clades A of  2*|A & F| / (|A| + |F|)

where the clades of an UNROOTED tree are both sides of every split. F1 = 1.0
means exact monophyly; F1 = 0.8 means the best-matching clade overlaps the
family substantially but leaks or omits members. Reported alongside is the
repair cost -- how many taxa must move for that family to become a clade --
which is the interpretable version of the same fact.

Equal weight per family is a deliberate choice, not a neutral one: it says a
14-language family counts as much as a 642-language one. Sizes are so skewed
here that any weighting is a value judgement; this one at least makes the
judgement explicit, where GQD's quartic weighting hides it. The size-stratified
table guards against the obvious failure mode, that the mean is carried by many
tiny easy families.

Runs anywhere -- pure stdlib, no ete3/qdist/R.

Usage:
    python3 code/family_recovery.py TREE [TREE...]
    python3 code/family_recovery.py data/phylip/ml_start_*.raxml.bestTree

Env:
    REF       reference tree            (default data/glottolog.tre)
    WORDLIST  Glottocode/Family source  (default data/lexibank_pruned_wordlist.csv)
    MIN_SIZE  smallest family to score  (default 4)
    OUT       write per-family CSV here (default: none)
"""
import argparse
import collections
import sys
from pathlib import Path

from src.jäger_scripts.analyze_tree_vs_glottolog import (  # noqa: E402
    family_map,
    leaf_names,
    split_masks,
)


def parse_args():
    p = argparse.ArgumentParser(
        description="Per-family recovery score: how nearly each Glottolog family is a clade"
    )
    p.add_argument("--trees", nargs="*", type=Path,
                   help="Inferred tree files to score")
    p.add_argument("--ref", default="data/glottolog.tre", type=Path,
                   help="Reference tree (default: data/glottolog.tre)")
    p.add_argument("--wordlist", default="data/lexibank_pruned_wordlist.csv", type=Path,
                   help="Glottocode/Family source (default: data/lexibank_pruned_wordlist.csv)")
    p.add_argument("--min-size", type=int, default=4,
                   help="Smallest family to score (default: 4)")
    p.add_argument("--out", default="", type=Path,
                   help="Write per-family CSV here (default: none)")
    return p.parse_args()


def clades(nw, name2bit, full):
    """Both sides of every non-trivial split, with sizes. Unrooted view."""
    out = {}
    for m in split_masks(nw, name2bit):
        if not (0 < m < full):
            continue
        for side in (m, full ^ m):
            if side.bit_count() >= 2:
                out[side] = side.bit_count()
    return sorted(out.items(), key=lambda kv: kv[1])


def best_f1(fam_mask, nf, cands):
    """Best F1 over candidate clades, plus that clade's overlap and size.

    The `ub` test is what makes this tractable: 2*min(na,nf)/(na+nf) bounds F1
    from above using only the sizes, so a candidate whose size is far from the
    family's is rejected without the expensive big-int AND.
    """
    best, best_ov, best_na = 0.0, 0, 0
    for A, na in cands:
        ub = 2.0 * min(na, nf) / (na + nf)
        if ub <= best:
            continue
        ov = (A & fam_mask).bit_count()
        f1 = 2.0 * ov / (na + nf)
        if f1 > best:
            best, best_ov, best_na = f1, ov, na
    return best, best_ov, best_na


def main():
    args = parse_args()

    ref_nw = args.ref.read_text()
    taxa = sorted(leaf_names(ref_nw))
    name2bit = {t: 1 << i for i, t in enumerate(taxa)}
    n_taxa = len(taxa)
    full = (1 << n_taxa) - 1

    gc2fam = family_map(args.wordlist)
    fam_n = collections.Counter()
    fam_mask = collections.defaultdict(int)
    for g, fam in gc2fam.items():
        if g in name2bit:
            fam_n[fam] += 1
            fam_mask[fam] |= name2bit[g]
    fams = sorted(
        (f for f, n in fam_n.items() if args.min_size <= n <= n_taxa - 2),
        key=lambda f: -fam_n[f],
    )
    print(f"reference: {args.ref}  ({n_taxa} taxa)")
    print(f"families scored: {len(fams)}  (size >= {args.min_size})\n")

    bands = [(4, 9), (10, 49), (50, 10 ** 9)]
    rows = {}
    for p in args.trees:
        label = p.name
        for suf in (".raxml.bestTree", ".raxml.lastTree.TMP", ".tre", ".nwk", ".txt"):
            label = label[: -len(suf)] if label.endswith(suf) else label
        label = label.replace("ml_start_", "")

        nw = p.read_text()
        if set(leaf_names(nw)) != set(taxa):
            print(f"ERROR: {label}: leaf set differs from reference", file=sys.stderr)
            return 1
        cands = clades(nw, name2bit, full)
        per = {}
        for f in fams:
            nf = fam_n[f]
            f1, ov, na = best_f1(fam_mask[f], nf, cands)
            # taxa to move: family members outside the clade, plus intruders in it
            per[f] = (f1, (nf - ov) + (na - ov))
        rows[label] = per

    order = list(rows)
    print("=== per-family F1, equal weight per family ===")
    hdr = f"  {'tree':10s} {'meanF1':>8s} {'exact':>7s} {'>=0.9':>7s} {'repair':>8s}   "
    hdr += " ".join(f"{a}-{b if b < 10**9 else '+':>3}".rjust(9) for a, b in bands)
    print(hdr)
    for k in order:
        per = rows[k]
        vals = [per[f][0] for f in fams]
        mean = sum(vals) / len(vals)
        exact = sum(1 for v in vals if v >= 0.999999)
        hi = sum(1 for v in vals if v >= 0.9)
        repair = sum(per[f][1] for f in fams)
        cells = []
        for a, b in bands:
            sel = [per[f][0] for f in fams if a <= fam_n[f] <= b]
            cells.append(f"{sum(sel) / len(sel):9.4f}" if sel else f"{'-':>9s}")
        print(f"  {k:10s} {mean:8.4f} {exact:7d} {hi:7d} {repair:8d}   " + " ".join(cells))

    if len(order) > 1:
        best = max(order, key=lambda k: sum(rows[k][f][0] for f in fams))
        print(f"\nbest mean F1: {best}")
    else:
        best = order[0]

    print(f"\n=== worst-recovered families ({best}) ===")
    per = rows[best]
    for f in sorted(fams, key=lambda f: per[f][0])[:12]:
        print(f"  {f:34s} n={fam_n[f]:4d}  F1={per[f][0]:.4f}  repair={per[f][1]:4d}")

    if args.out:
        with args.out.open("w") as fh:
            fh.write("tree,family,n,f1,repair\n")
            for k in order:
                for f in fams:
                    f1, rep = rows[k][f]
                    fh.write(f'{k},"{f}",{fam_n[f]},{f1:.6f},{rep}\n')
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
