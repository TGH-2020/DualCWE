"""
Infer per-family trees from a distance matrix and compute Generalized
Quartet Distance (GQD) for each family against its pruned Glottolog tree.
Note that this has been ran on Windows. When running on Linux, you can remove the WSL calls and just run `qdist` directly.

Workflow:
    1. Run src/get_family_trees.R  ->  per-family NJ/UPGMA + Glottolog trees
    2. For each family with a Glottolog tree, run `wsl qdist`
    3. Save results to CSV, sorted by family size (descending)

Requires: R (with ape, phangorn), WSL with qdist installed.

Inputs:
    --dist_csv: language distance matrix CSV; required
    --forms_csv: lexibank_pruned_wordlist.csv (for family membership); default=data/lexibank_pruned_wordlist.csv
    --glottolog: Glottolog tree file; default=data/glottolog.tre
    --out_dir: directory for family tree files; default=out/families/
    --results: output CSV for GQD results; required
    --r_script: R script that generates the family trees; default=src/get_family_trees.R
    --min_languages: minimum family size to keep; default=4
    --skip_r: skip R tree generation and use existing files in --out_dir

Outputs:
    results: CSV with per-family GQD results
    family_gqd_summary.csv: summary of GQD by tree type and family size bin

Example:
    python -m src.get_qdist_families \\
        --dist_csv    out/cosine_dist.csv \\
        --forms_csv   data/lexibank_pruned_wordlist.csv \\
        --glottolog   data/glottolog.tre \\
        --out_dir     out/families/ \\
        --results     results/family_gqd_results.csv
"""

import argparse
import re
import subprocess
from collections import Counter
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# Utilities
def to_wsl_path(windows_path) -> str:
    p = str(windows_path).replace("\\", "/")
    if len(p) > 1 and p[1] == ":":
        p = "/mnt/" + p[0].lower() + p[2:]
    return p


def run_qdist_wsl(tree1_path: Path, tree2_path: Path):
    """
    Run `qdist` via WSL and return GQD = 1 - normalised_shared_butterflies.
    Returns None on error.
    """
    try:
        proc = subprocess.run(
            ["wsl", "-d", "Ubuntu", "qdist",
             to_wsl_path(tree1_path), to_wsl_path(tree2_path)],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            print(f"  qdist stderr: {proc.stderr.strip()}")
            return None
        parts = proc.stdout.strip().split()
        if len(parts) >= 16:
            return 1.0 - float(parts[15])
        print(f"  Unexpected qdist output: {proc.stdout.strip()}")
        return None
    except Exception as e:
        print(f"  Error running qdist: {e}")
        return None


def safe_name(name: str) -> str:
    """Sanitise a string to match the R filename convention."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def run_family_tree_generation(
    r_script, dist_csv, forms_csv, glottolog, out_dir, min_languages
) -> None:
    subprocess.run(
        ["Rscript", str(r_script),
         "--dist_csv",       str(dist_csv),
         "--forms_csv",      str(forms_csv),
         "--glottolog",      str(glottolog),
         "--out_dir",        str(out_dir),
         "--min_languages",  str(min_languages)],
        check=True, text=True,
    )


def build_family_info(dist_csv: str, forms_csv: str) -> dict:
    """
    Return a dict mapping safe_name(family) -> {family, n_languages} using only
    languages present in the distance matrix.
    """
    dist_langs = set(pd.read_csv(dist_csv, index_col=0).index)
    forms = pd.read_csv(forms_csv)
    forms = (
        forms[forms["Glottocode"].isin(dist_langs)]
        .drop_duplicates(subset="Glottocode")
    )
    counts = Counter(forms["Family"].dropna())
    return {
        safe_name(fam): {"family": fam, "n_languages": n}
        for fam, n in counts.items()
    }


def summarize_gqd(df: pd.DataFrame) -> pd.DataFrame:
    """Mean/sd of GQD per tree type, overall and binned by family size."""
    bin_edges = [3, 10, 20, 50, float("inf")]
    bin_labels = ["4-10", "11-20", "21-50", "51+"]
    df = df.assign(size_bin=pd.cut(df["n_languages"], bins=bin_edges, labels=bin_labels))

    overall = df.groupby("tree_type")["GQD"].agg(mean="mean", sd="std", n_families="count").reset_index()
    overall.insert(1, "size_bin", "all")

    by_bin = (
        df.groupby(["tree_type", "size_bin"], observed=True)["GQD"]
        .agg(mean="mean", sd="std", n_families="count")
        .reset_index()
    )

    return pd.concat([overall, by_bin], ignore_index=True)


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Compute per-family GQD")
    p.add_argument("--dist_csv",   required=True, help="Language distance matrix CSV")
    p.add_argument("--forms_csv",  default="data/lexibank_pruned_wordlist.csv",
                   help="lexibank_pruned_wordlist.csv (for family membership)")
    p.add_argument("--glottolog",  default="data/glottolog.tre")
    p.add_argument("--out_dir",    default="out/families/", help="Directory for family tree files")
    p.add_argument("--results",    required=True, help="Output CSV for GQD results")
    p.add_argument("--r_script",   default="src/get_family_trees.R")
    p.add_argument("--min_languages", type=int, default=4)
    p.add_argument("--skip_r", action="store_true",
                   help="Skip R tree generation and use existing files in --out_dir")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    if not args.skip_r:
        print("Running get_family_trees.R ...")
        run_family_tree_generation(
            args.r_script, args.dist_csv, args.forms_csv,
            args.glottolog, out_dir, args.min_languages,
        )

    # Build family size lookup
    family_info = build_family_info(args.dist_csv, args.forms_csv)

    # Match inferred trees to their Glottolog counterparts
    glottolog_trees = {
        f.stem[: -len("_glottolog")]: f
        for f in out_dir.glob("*_glottolog.txt")
    }
    inferred_trees = sorted(
        f for f in out_dir.glob("*.txt") if "_glottolog" not in f.name
    )

    if not inferred_trees:
        print("No inferred family tree files found in", out_dir)
        return

    results = []
    for f in tqdm(inferred_trees, desc="Computing family GQD"):
        # filename format: <family_safe>_<nj|upgma>.txt
        parts = f.stem.rsplit("_", 1)
        if len(parts) != 2 or parts[1] not in ("nj", "upgma"):
            continue
        fam_safe, tree_type = parts

        glottolog_tree = glottolog_trees.get(fam_safe)
        if glottolog_tree is None:
            print(f"  No Glottolog tree for '{fam_safe}', skipping.")
            continue

        gqd = run_qdist_wsl(f, glottolog_tree)
        if gqd is None or pd.isna(gqd):
            print(f"  Skipping {f.name}")
            continue

        info = family_info.get(fam_safe, {})
        results.append({
            "family":       info.get("family", fam_safe),
            "n_languages":  info.get("n_languages"),
            "tree_type":    tree_type,
            "GQD":          gqd,
        })

    if not results:
        print("No results produced.")
        return

    df = (
        pd.DataFrame(results)
        .sort_values(["n_languages", "family", "tree_type"],
                     ascending=[False, True, True])
    )
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nFamily GQD results saved to {out}")
    print(df.to_string(index=False))

    summary = summarize_gqd(df)

    print("\nSummary by tree type:")
    print(summary.to_string(index=False))
    out_summary = out.parent / "family_gqd_summary.csv"
    summary.to_csv(out_summary, index=False)


if __name__ == "__main__":
    main()
