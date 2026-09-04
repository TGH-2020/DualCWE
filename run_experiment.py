"""
Run the full experiment pipeline across N seeds.
You can provide your own model and as long as it follows the expected interface, it can be used in the experiment pipeline.

Per seed, the pipeline runs:
    1. src.train: 
        outputs config and model to 
            checkpoints/seed{s}/
    2. src.evaluate_pairwise_distances: 
        outputs distances and embeddings to 
            out/seed{s}/cosine_dist.csv
            out/seed{s}/embeddings.npz
    3. src.get_qdist_full:
        outputs inferred trees and the GQD results to
            out/seed{s}/
            results/seed{s}/full_gqd_results.csv
    4. src.get_qdist_families:
        outputs inferred trees and the GQD results for language families to
            out/seed{s}/families/
            results/seed{s}/family_gqd_results.csv

After all seeds, summary CSVs (mean +/- SD across seeds) are written to:
    results/full_gqd_summary.csv
    results/family_gqd_summary.csv

Inputs:
    --seeds: random seeds to use; default=1 2 3 4 5
    --forms_path: path to lexibank_pruned_wordlist.csv; default=data/lexibank_pruned_wordlist.csv
    --concepts_path: path to dellert_ranking.csv; default=data/dellert_ranking.csv
    --token_type: segment representation to use (ipa, sca, dolgo, asjp, featvecs); default=featvecs
    --model_type: model architecture to train; default=dual_contrastive
    --model_name: model name prefix; default=dualcwe
    --to_rank: keep only the top-N ranked concepts; default=all
    --batch_size: batch size; default=512
    --num_epochs: number of training epochs; default=15
    --embedding_dim: embedding dimension; default=256
    --encoder_layers: number of encoder layers; default=1
    --learning_rate: learning rate; default=1e-3
    --temperature: temperature for contrastive loss; default=0.3
    --swap_p: probability of swapping two characters in a word; default=0.0
    --dup_p: probability of duplicating a character in a word; default=0.1
    --noise_p: probability of adding noise to a character in a word; default=0.5
    --del_p: probability of deleting a character in a word; default=0.0
    --input_dropout: dropout probability for input tokens; default=0.1
    --num_langs_per_batch: number of languages per batch for dual contrastive learning; default=32
    --num_concepts_per_lang: number of concepts per language for dual contrastive learning; default=30
    --lang_loss_weight: weight for the language-level contrastive loss; default=0.2
    --eval_epoch: epoch at which to evaluate; default=last epoch
    --glottolog: Glottolog tree file; default=data/glottolog.tre
    --out_dir: base directory for per-seed distance/tree outputs; default=out
    --results_dir: base directory for per-seed and summary results; default=results
    --checkpoints_dir: base directory for per-seed model checkpoints; default=checkpoints
    --model_prefix: prefix for model/checkpoint names (appended with seed); default=seed
    --skip_train: skip training; use existing checkpoints
    --skip_distances: skip distance computation; use existing CSVs
    --skip_trees: skip tree inference; use existing tree files
    --skip_families: skip per-family analysis; default=True
    --min_languages: minimum family size for per-family tree inference; default=4

Outputs:
    full_gqd_summary.csv: mean +/- SD of full-language GQD per tree type
    family_gqd_summary.csv: mean +/- SD of per-family GQD
    full_gqd_all_seeds.csv: combined per-seed full GQD results
    family_gqd_all_seeds.csv: combined per-seed family GQD results

Example:
    python run_experiment.py # runs the full experiment with default settings
    python run_experiment.py --token_type dolgo --model_type your_model # runs a model you added with characters encoded as Dolgopolsky sound classes
    python run_experiment.py --skip_train --skip_distances  # only re-run trees
    python run_experiment.py --skip_train --eval_epoch 5 # skip training and evaluate the specified epoch
"""

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


# Helpers

def run_module(module: str, module_args: list, description: str) -> None:
    """Run `python -m <module> <module_args>` as a subprocess."""
    cmd = [sys.executable, "-m", module] + [str(a) for a in module_args]
    print(f"\n{'=' * 60}")
    print(f"  {description}")
    print(f"  {' '.join(cmd)}")
    print(f"{'=' * 60}")
    subprocess.run(cmd, check=True)


def summarise(combined: pd.DataFrame, group_cols: list) -> pd.DataFrame:
    """Compute mean and SD of GQD over seeds, grouped by group_cols."""
    return (
        combined.groupby(group_cols)["GQD"]
        .agg(mean_GQD="mean", sd_GQD="std")
        .reset_index()
    )


# CLI

def parse_args():
    p = argparse.ArgumentParser(
        description="Run the full experiment pipeline across multiple seeds"
    )

    # Experiment
    p.add_argument("--seeds", type=int, nargs="+", default=[1,2,3,4,5],
                   help="Random seeds to use (default: 1 2 3 4 5)")

    # Data / training (forwarded to src.train)
    p.add_argument("--forms_path",    default="data/lexibank_pruned_wordlist.csv")
    p.add_argument("--concepts_path", default="data/dellert_ranking.csv")
    p.add_argument("--token_type",    default="featvecs",
                   choices=["ipa", "sca", "dolgo", "asjp", "featvecs"])
    p.add_argument("--model_type",    default="dual_contrastive",
                   help="Model type to train (forwarded to src.train)")
    p.add_argument("--model_name",    default="dualcwe")
    p.add_argument("--to_rank",       type=int, default=None)
    p.add_argument("--batch_size",    type=int, default=512)
    p.add_argument("--num_epochs",    type=int, default=15)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--encoder_layers", type=int, default=1)
    p.add_argument("--learning_rate",   type=float, default=1e-3)

    # Training - contrastive learning
    p.add_argument("--temperature", type=float, default=0.3,
                   help="Temperature for contrastive loss")
    p.add_argument("--swap_p", type=float, default=0.0,
                   help="Probability of swapping two characters in a word for contrastive learning")
    p.add_argument("--dup_p", type=float, default=0.1,
                   help="Probability of duplicating a character in a word for contrastive learning")
    p.add_argument("--noise_p", type=float, default=0.5,
                   help="Probability of adding noise to a character in a word for contrastive learning")
    p.add_argument("--del_p", type=float, default=0.0,
                   help="Probability of deleting a character in a word for contrastive learning")
    p.add_argument("--input_dropout", type=float, default=0.1,
                   help="Dropout probability for input tokens in contrastive learning")
    p.add_argument("--num_langs_per_batch", type=int, default=32,
                   help="Number of languages per batch for dual contrastive learning")
    p.add_argument("--num_concepts_per_lang", type=int, default=30,
                   help="Number of concepts per language for dual contrastive learning")
    p.add_argument("--lang_loss_weight", type=float, default=0.2,
                   help="Weight for the language-level contrastive loss in dual contrastive learning. Set to 0 to disable language-level loss.")

    # Eval (forwarded to src.evaluate_pairwise_distances)
    p.add_argument("--eval_epoch", type=int, default=None,
                   help="Epoch at which to evaluate (default: last epoch)")

    # Paths
    p.add_argument("--glottolog",      default="data/glottolog.tre")
    p.add_argument("--out_dir",        default="out",
                   help="Base directory for per-seed distance/tree outputs")
    p.add_argument("--results_dir",    default="results",
                   help="Base directory for per-seed and summary results")
    p.add_argument("--checkpoints_dir",default="checkpoints",
                   help="Base directory for per-seed model checkpoints")
    p.add_argument("--model_prefix",  default="seed",
                   help="Prefix for model/checkpoint names (appended with seed)")

    # Skip flags (for re-running only later pipeline stages)
    p.add_argument("--skip_train",     action="store_true",
                   help="Skip training; use existing checkpoints")
    p.add_argument("--skip_distances", action="store_true",
                   help="Skip distance computation; use existing CSVs")
    p.add_argument("--skip_trees",     action="store_true",
                   help="Skip tree inference; use existing tree files")
    p.add_argument("--skip_families",  action="store_true", default=True,
                   help="Skip per-family analysis")
    p.add_argument("--min_languages",  type=int, default=4,
                   help="Minimum family size for per-family tree inference")

    return p.parse_args()


# Main

def main():
    args = parse_args()

    all_full_results   = []
    all_family_results = []

    for seed in tqdm(args.seeds, desc="Running seeds"):
        tag         = f"{args.model_prefix}{seed}"
        ckpt_dir    = Path(args.checkpoints_dir) / tag
        out_seed    = Path(args.out_dir)     / tag
        res_seed    = Path(args.results_dir) / tag
        model_name  = f"{args.model_name}_{tag}"
        config_path = ckpt_dir / f"{model_name}_config.json"
        dist_csv    = out_seed / "cosine_dist.csv"
        emb_path    = out_seed / "embeddings.npz"
        fam_dir     = out_seed / "families"
        full_res    = res_seed / "full_gqd_results.csv"
        fam_res     = res_seed / "family_gqd_results.csv"

        #############################
        ######## 1. Train ###########
        #############################
        if not args.skip_train:
            train_args = [
                "--model_type",    args.model_type,
                "--forms_path",    args.forms_path,
                "--concepts_path", args.concepts_path,
                "--token_type",    args.token_type,
                "--batch_size",    args.batch_size,
                "--num_epochs",    args.num_epochs,
                "--embedding_dim", args.embedding_dim,
                "--encoder_layers", args.encoder_layers,
                "--seed",          seed,
                "--model_name",    model_name,
                "--checkpoint_dir",ckpt_dir,
                "--temperature",     args.temperature,
                "--swap_p",          args.swap_p,
                "--dup_p",           args.dup_p,
                "--noise_p",         args.noise_p,
                "--del_p",           args.del_p,
                "--input_dropout",   args.input_dropout,
                "--learning_rate",     args.learning_rate,
                "--num_langs_per_batch", args.num_langs_per_batch,
                "--num_concepts_per_lang", args.num_concepts_per_lang,
            ]
            if args.to_rank is not None:
                train_args += ["--to_rank", args.to_rank]
            run_module("src.train", train_args, f"[Seed {seed}] Train")

        #############################
        ### 2. Pairwise distances ###
        #############################
        if not args.skip_distances:
            dist_args = [
                "--config",         config_path,
                "--output",         dist_csv,
                "--save_embeddings",emb_path,
            ]
            if args.eval_epoch is not None:
                dist_args += ["--eval_epoch", args.eval_epoch]
            run_module("src.evaluate_pairwise_distances", dist_args,
                       f"[Seed {seed}] Pairwise distances")

        #############################
        ###### 3. Full GQD ##########
        #############################
        if not args.skip_trees:
            full_args = [
                "--dist_csv",  dist_csv,
                "--glottolog", args.glottolog,
                "--out_dir",   out_seed,
                "--results",   full_res,
            ]
            run_module("src.get_qdist_full", full_args,
                       f"[Seed {seed}] Full GQD")

        #############################
        ##### 4. Family GQD #########
        #############################
        if not args.skip_trees and not args.skip_families:
            fam_args = [
                "--dist_csv",       dist_csv,
                "--forms_csv",      args.forms_path,
                "--glottolog",      args.glottolog,
                "--out_dir",        fam_dir,
                "--results",        fam_res,
                "--min_languages",  args.min_languages,
            ]
            run_module("src.get_qdist_families", fam_args,
                       f"[Seed {seed}] Family GQD")

        #############################
        ##### Collect results #######
        #############################
        if full_res.exists():
            df = pd.read_csv(full_res)
            df["seed"] = seed
            all_full_results.append(df)

        if fam_res.exists():
            df = pd.read_csv(fam_res)
            df["seed"] = seed
            all_family_results.append(df)

    # Summary statistics
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  Summary statistics")
    print(f"{'=' * 60}")

    if all_full_results:
        combined = pd.concat(all_full_results, ignore_index=True)

        # Per-seed combined (for reference)
        combined.to_csv(results_dir / "full_gqd_all_seeds.csv", index=False)

        # Summary: mean +/- SD per tree_type
        summary = summarise(combined, ["tree_type"]).sort_values("mean_GQD")
        summary.to_csv(results_dir / "full_gqd_summary.csv", index=False)

        print("\nFull-language GQD  (lower = better match with Glottolog)")
        print(summary.to_string(index=False))
        print(f"\n  Saved to {results_dir / 'full_gqd_summary.csv'}")
    else:
        print("\nNo full GQD results found.")

    if all_family_results:
        combined_fam = pd.concat(all_family_results, ignore_index=True)

        # Per-seed combined
        combined_fam.to_csv(results_dir / "family_gqd_all_seeds.csv", index=False)

        # Summary: mean +/- SD per family x tree_type, sorted by family size
        sum_cols = [c for c in ["family", "n_languages", "tree_type"]
                    if c in combined_fam.columns]
        summary_fam = summarise(combined_fam, sum_cols)
        if "n_languages" in summary_fam.columns:
            summary_fam = summary_fam.sort_values(
                ["n_languages", "family", "tree_type"],
                ascending=[False, True, True],
            )
        summary_fam.to_csv(results_dir / "family_gqd_summary.csv", index=False)

        print("\nPer-family GQD  (sorted by family size, lower = better)")
        print(summary_fam.to_string(index=False))
        print(f"\n  Saved to {results_dir / 'family_gqd_summary.csv'}")
    else:
        print("\nNo family GQD results found.")


if __name__ == "__main__":
    main()
