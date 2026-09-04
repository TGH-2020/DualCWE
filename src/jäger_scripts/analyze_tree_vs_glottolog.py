#!/usr/bin/env python3
"""
Structural comparison of inferred trees against the Glottolog classification.

GQD (code/compute_gqd.sh) reduces a tree to one number. That number turned out
to rank trees *opposite* to every split-based measure (see STATUS.md, "What GQD
actually weights here"), so this script provides the structural measures needed
to interpret a GQD difference rather than take it at face value:

  1. family monophyly      -- how many families are recovered as clades
  2. pairwise RF           -- how the trees relate to each other and to Glottolog
  3. quartet composition   -- what kinds of quartet Glottolog actually resolves,
                              which is what GQD is implicitly weighting

Finding (3) is the important one: `glottolog.tre` is essentially a star of
families, so ~87 % of its resolved quartets are of the form "a pair from one
family vs. two taxa from two other families". GQD therefore measures the
cohesion of same-family *pairs*, not family monophyly and not within-family
resolution -- and those pairs are ~78 % concentrated in Atlantic-Congo and
Austronesian. Small GQD gaps are not a robust "closer to Glottolog" signal.

Runs anywhere -- pure stdlib, no ete3/qdist/R needed. Trees are compared as
UNROOTED: a family counts as recovered if its taxa form one side of a split.

Usage:
    python3 code/analyze_tree_vs_glottolog.py TREE [TREE...]
    python3 code/analyze_tree_vs_glottolog.py data/phylip/ml_start_p*.raxml.bestTree

Env:
    REF       reference tree            (default data/glottolog.tre)
    WORDLIST  Glottocode/Family source  (default data/lexibank_pruned_wordlist.csv)
"""
import argparse
import collections
import csv
import sys
from math import comb
from pathlib import Path

def parse_args():
    p = argparse.ArgumentParser(
        description="Structural comparison of inferred trees against the Glottolog classification"
    )
    p.add_argument("trees", nargs="*", type=Path,
                   help="Inferred tree files to compare against the reference")
    p.add_argument("--ref", default="data/glottolog.tre", type=Path,
                   help="Reference tree (default: data/glottolog.tre)")
    p.add_argument("--wordlist", default="data/lexibank_pruned_wordlist.csv", type=Path,
                   help="Glottocode/Family source (default: data/lexibank_pruned_wordlist.csv)")
    return p.parse_args()


def _scan(nw, on_leaf, on_close):
    """Single-pass Newick walk. Calls on_leaf(name) / on_close().

    Labels and branch lengths that follow a ')' belong to the internal node just
    closed, not to a new leaf, so they are discarded via the last_closed flag.
    """
    buf = []
    last_closed = [False]

    def flush():
        s = "".join(buf).strip()
        buf.clear()
        if last_closed[0]:
            last_closed[0] = False
            return
        if not s:
            return
        name = s.split(":")[0].strip()
        if name:
            on_leaf(name)

    for c in nw:
        if c == "(":
            flush()
            on_close(open_=True)
        elif c == ",":
            flush()
        elif c == ")":
            flush()
            on_close(open_=False)
            last_closed[0] = True
        elif c == ";":
            flush()
            break
        else:
            buf.append(c)


def leaf_names(nw):
    out = []
    _scan(nw, out.append, lambda open_: None)
    return out


def split_masks(nw, name2bit):
    """Every internal node's leaf-set, as an int bitmask."""
    stack = [[]]
    masks = []

    def on_leaf(name):
        stack[-1].append(name2bit[name])

    def on_close(open_):
        if open_:
            stack.append([])
            return
        kids = stack.pop()
        m = 0
        for k in kids:
            m |= k
        masks.append(m)
        stack[-1].append(m)

    _scan(nw, on_leaf, on_close)
    return masks


def family_map(path):
    """Glottocode -> family, by majority non-empty label.

    Necessary because the wordlist merges several source databases: 137
    glottocodes carry more than one Family value, nearly all of them an empty
    string in one db and a real family in another, plus a few genuine
    reclassifications (e.g. hrus1242 Hruso/Sino-Tibetan).
    """
    votes = collections.defaultdict(collections.Counter)
    with path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g, fam = row["Glottocode"], (row["Family"] or "").strip()
            if g and fam:
                votes[g][fam] += 1
    return {g: c.most_common(1)[0][0] for g, c in votes.items()}


def main():
    args = parse_args()

    ref_nw = args.ref.read_text()
    taxa = sorted(leaf_names(ref_nw))
    name2bit = {t: 1 << i for i, t in enumerate(taxa)}
    n_taxa = len(taxa)
    full = (1 << n_taxa) - 1
    print(f"reference: {args.ref}  ({n_taxa} taxa)")

    trees = {"glottolog": ref_nw}
    for p in args.trees:
        label = p.name
        for suf in (".raxml.bestTree", ".raxml.lastTree.TMP", ".tre", ".nwk", ".txt"):
            label = label[: -len(suf)] if label.endswith(suf) else label
        label = label.replace("ml_start_", "")
        trees[label] = p.read_text()

    def canonical(masks):
        """Non-trivial splits, oriented so bit 0 is clear -- makes them comparable."""
        out = set()
        for m in masks:
            pc = m.bit_count()
            if pc < 2 or pc > n_taxa - 2:
                continue
            out.add(m if not (m & 1) else full ^ m)
        return out

    raw, canon = {}, {}
    for k, nw in trees.items():
        got = set(leaf_names(nw))
        if got != set(taxa):
            print(f"ERROR: {k}: leaf set differs from reference "
                  f"({len(got)} vs {n_taxa})", file=sys.stderr)
            return 1
        masks = split_masks(nw, name2bit)
        raw[k] = {m for m in masks if 0 < m < full} | {
            full ^ m for m in masks if 0 < m < full
        }
        canon[k] = canonical(masks)

    order = [k for k in trees if k != "glottolog"]

    gc2fam = family_map(args.wordlist)
    fam_n = collections.Counter()
    fam_mask = collections.defaultdict(int)
    for g, fam in gc2fam.items():
        if g in name2bit:
            fam_n[fam] += 1
            fam_mask[fam] |= name2bit[g]
    testable = {f: n for f, n in fam_n.items() if 3 <= n <= n_taxa - 2}

    print(f"families represented: {len(fam_n)}   testable: {len(testable)}\n")

    print("=== family monophyly (unrooted: family forms one side of a split) ===")
    mono = {}
    for k in ["glottolog"] + order:
        mono[k] = {f for f in testable if fam_mask[f] in raw[k]}
        print(f"  {k:12s} {len(mono[k]):4d} / {len(testable)}")
    print("  (glottolog falls short of 153 only because the family labels come "
          "from the\n   wordlist, not from the tree; a handful disagree.)")

    print("\n=== splits shared with Glottolog ===")
    for k in order:
        print(f"  {k:12s} shared={len(canon[k] & canon['glottolog']):5d}"
              f"   RF={len(canon[k] ^ canon['glottolog'])}")

    if len(order) > 1:
        print("\n=== pairwise RF ===")
        print("            " + "".join(f"{o:>8s}" for o in order))
        for a in order:
            print(f"  {a:>9s} " + "".join(f"{len(canon[a] ^ canon[b]):8d}"
                                          for b in order))

    print("\n=== what Glottolog actually resolves (drives GQD) ===")
    deep = sum(
        1
        for m in split_masks(ref_nw, name2bit)
        if m.bit_count() < n_taxa - 1
        and sum(1 for f, M in fam_mask.items() if M and (M & m) == M) > 1
    )
    print(f"  glottolog splits grouping >=2 whole families: {deep}"
          "  (~0 => it is a star of families)")
    sizes = sorted(fam_n.values(), reverse=True)
    tot = sum(sizes)
    s_pairs = sum(comb(n, 2) for n in sizes)
    q22 = (s_pairs**2 - sum(comb(n, 2) ** 2 for n in sizes)) // 2
    q211 = sum(
        comb(n, 2) * (comb(tot - n, 2) - (s_pairs - comb(n, 2))) for n in sizes
    )
    print(f"  same-family pairs           {s_pairs:>18,}")
    print(f"  Q(2+2)  two families        {q22:>18,}")
    print(f"  Q(2+1+1) three families     {q211:>18,}   <- dominant term")
    print(f"  subtotal                    {q22 + q211:>18,}")
    print("  => GQD mostly measures whether same-family PAIRS stay separated "
          "from outsiders.")
    top = sorted(fam_n.items(), key=lambda x: -comb(x[1], 2))[:4]
    for f, n in top:
        print(f"     {f:28s} n={n:4d}  {100 * comb(n, 2) / s_pairs:5.1f} % of pairs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
