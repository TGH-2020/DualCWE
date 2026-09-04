"""
Infer full-language trees from a distance matrix and compute Generalized
Quartet Distance (GQD) against the pruned Glottolog tree.
Note that this has been ran on Windows. When running on Linux, you can remove the WSL calls and just run `qdist` directly.

Workflow:
    1. Run src/get_full_trees.R  ->  saves NJ/UPGMA trees + pruned Glottolog tree
    2. For each inferred tree, run `wsl qdist <tree> <glottolog>`
    3. Save GQD results to CSV

Requires: R (with ape, phangorn), WSL with qdist installed.

Inputs:
    --dist_csv: language distance matrix CSV; required unless --skip_r is set
    --glottolog: Glottolog tree file; default=data/glottolog.tre
    --out_dir: directory for tree files; default=out/
    --results: output CSV for GQD results; required
    --r_script: R script that generates the full trees; default=src/get_full_trees.R
    --skip_r: skip tree generation and use existing files in --out_dir

Outputs:
    results: CSV with per-tree GQD results

Example:
    python -m src.get_qdist_full \\
        --dist_csv  out/cosine_dist.csv \\
        --glottolog data/glottolog.tre \\
        --out_dir   out/ \\
        --results   results/full_gqd_results.csv
"""

import argparse
import subprocess
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

    qdist output field 15 (0-indexed) is the normalised shared butterfly
    count.
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


def run_tree_generation(r_script: Path, dist_csv: Path, glottolog: Path, out_dir: Path) -> None:
    subprocess.run(
        ["Rscript", str(r_script),
         "--dist_csv",  str(dist_csv),
         "--glottolog", str(glottolog),
         "--out_dir",   str(out_dir)],
        check=True, text=True,
    )


# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Infer full trees and compute GQD")
    p.add_argument("--dist_csv",  default=None, help="Language distance matrix CSV")
    p.add_argument("--glottolog", default="data/glottolog.tre")
    p.add_argument("--out_dir",   default="out/", help="Directory for tree files")
    p.add_argument("--results",   required=True, help="Output CSV for GQD results")
    p.add_argument("--r_script",  default="src/get_full_trees.R")
    p.add_argument("--skip_r", action="store_true",
                   help="Skip tree generation and use existing files in --out_dir")
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)

    if not args.skip_r:
        if args.dist_csv is None:
            raise ValueError("Must provide --dist_csv unless --skip_r is set")
        print("Running get_full_trees.R ...")
        run_tree_generation(Path(args.r_script), Path(args.dist_csv),
                            Path(args.glottolog), out_dir)

    glottolog_tree = out_dir / "pruned_glottolog_tree.txt"
    if not glottolog_tree.exists():
        raise FileNotFoundError(f"Pruned Glottolog tree not found: {glottolog_tree}")

    tree_files = sorted(out_dir.glob("inferred_tree_*.txt"))
    if not tree_files:
        print("No inferred tree files found in", out_dir)
        return

    results = []
    for f in tqdm(tree_files, desc="Computing GQD"):
        # filename format: inferred_tree_<dist_name>_<nj|upgma>.txt
        tree_type = f.stem.rsplit("_", 1)[-1]  # "nj" or "upgma"
        gqd = run_qdist_wsl(f, glottolog_tree)
        if gqd is None:
            print(f"  Skipping {f.name}")
            continue
        results.append({"tree_file": f.name, "tree_type": tree_type, "GQD": gqd})

    if not results:
        print("No results produced.")
        return

    df = pd.DataFrame(results).sort_values("GQD")
    out = Path(args.results)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nGQD results saved to {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
