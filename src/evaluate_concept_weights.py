"""
Compute per-concept stability weights and correlate with external rankings.

Inputs:
    --config: path to the training configuration JSON file; required
    --epoch: epoch number to use when computing embeddings from scratch; optional, default is the latest checkpoint.
    --embeddings: path to pre-computed embeddings file (.npz) produced by evaluate_pairwise_distances.py --save_embeddings; optional, if not provided, embeddings will be computed from scratch using --config and --epoch.
    --rankings: path to external rankings file (TSV); optional
    --output_dir: directory to save the output CSV files; required
    --block: block size for blockwise computation; optional, default is 256. Reduce this if you encounter memory issues.

Outputs:
    concept_weights.csv: per-concept weights and ranks
    stability_correlations.csv: Spearman correlations with external rankings (only when --rankings is provided)

Example:
    python -m src.evaluate_concept_weights \\
        --config checkpoints/dualcwe_config.json \\
        --rankings data/mapped-and-ranked-lists.tsv \\
        --output_dir results/
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader
from tqdm import trange

from src.config import TrainConfig, build_model
from src.data_prep import LexicalDataset, collate_fn
from src.evaluate_pairwise_distances import get_mean_embeddings
from src.training_util import load_model_checkpoint

RANKINGS = {
    "Holman ranking value": True,        # higher = more stable
    "Pagel mean rate": False,            # lower = more stable
    "Petroni & Serva": True,             # higher = more stable
    "Peust ranking": False,              # lower = more stable
    "Rama & Borin score": False,         # lower = more stable
    "Tadmor score": True,                # higher = more stable
    "Tadmor score (replica)": True,      # higher = more stable
    "WOLD age score": True,              # higher = more stable
    "WOLD borrowed score": False,        # lower = more stable
    "WOLD simplicity score": True,       # higher = more stable
    "inf": False,                        # lower = more stable
}

def clean_concepts(name: str) -> str:
    return "".join(c for c in name if c.isalnum()).lower()

def load_rankings(rankings_path):
    """
    Read the external stability rankings from Dellert's 2018 study.

    Returns:
        rankings: dict mapping ranking name to a dict mapping each cleaned concept name to its score in that ranking.
    """
    rankings = defaultdict(dict)
    with open(rankings_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            concept = clean_concepts(row.get("concepticon", ""))
            if not concept:
                continue
            for col in RANKINGS.keys():
                val = row.get(col, "").strip()
                if val:
                    try:
                        rankings[col][concept] = float(val)
                    except ValueError:
                        pass
    return rankings


def compute_concept_weights(mean_emb, valid, block=256):
    """
    Per-concept stability scores: variance of D_c.
    Concepts with higher variance are considered more stable. A concept needs to be included in at least L/4 languages to be considered; otherwise, it is assigned inf.

    Args:
        mean_emb: (L, C, D) float32 tensor
        valid: (L, C) bool mask
        block: block size for blockwise computation

    Returns:
        weights: dict mapping c to (1-variance) (lower = more stable; inf = excluded)
    """
    device = mean_emb.device
    L, C, _ = mean_emb.shape
    normed = mean_emb / mean_emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)

    # Check which language pairs share a concept
    dist_cnt = torch.zeros(L, L, dtype=torch.int64, device=device)

    for i0 in trange(0, L, block, desc="Counting shared concepts"):
        i1 = min(i0 + block, L)
        Av = valid[i0:i1]
        for j0 in range(0, L, block):
            j1 = min(j0 + block, L)
            Bv = valid[j0:j1]
            sim = torch.einsum("acd,bcd->abc", normed[i0:i1], normed[j0:j1])
            dist = (1.0 - sim).double()
            vm = Av.unsqueeze(1) & Bv.unsqueeze(0)
            dist_cnt[i0:i1, j0:j1] += vm.sum(2)

    pair_valid = dist_cnt > 0  # exclude pairs with no shared concept

    # per-concept std over related pairs
    n_c  = torch.zeros(C, dtype=torch.float64, device=device)
    s_d  = torch.zeros(C, dtype=torch.float64, device=device)
    s_d2 = torch.zeros(C, dtype=torch.float64, device=device)

    for i0 in trange(0, L, block, desc="Calculating concept variance"):
        i1 = min(i0 + block, L)
        Av = valid[i0:i1]
        for j0 in range(i0, L, block):
            j1 = min(j0 + block, L)
            Bv = valid[j0:j1]
            sim = torch.einsum("acd,bcd->abc", normed[i0:i1], normed[j0:j1])
            dist = (1.0 - sim).double()
            vm = (Av.unsqueeze(1) & Bv.unsqueeze(0)) & pair_valid[i0:i1, j0:j1].unsqueeze(2)
            if i0 == j0:
                tri = torch.triu(
                    torch.ones(i1 - i0, j1 - j0, dtype=torch.bool, device=device),
                    diagonal=1,
                ).unsqueeze(2)
                vm = vm & tri
            vm_d = vm.double()
            n_c  += vm_d.sum(dim=(0, 1))
            s_d  += (dist * vm_d).sum(dim=(0, 1))
            s_d2 += (dist ** 2 * vm_d).sum(dim=(0, 1))

    coverage = valid.sum(dim=0)
    min_coverage = L // 4
    weights = {}
    for c in range(C):
        if coverage[c].item() < min_coverage:
            weights[c] = float("inf")
            continue
        n = n_c[c].item()
        if n < 2:
            weights[c] = float("inf")
            continue
        var = s_d2[c].item() / n - (s_d[c].item() / n) ** 2
        weights[c] = 1 - float(np.sqrt(max(var, 0.0)))

    return weights


# Stability ranking correlations
def stability_ranking_correlations(concept_weights, conc2idx, rankings):
    """
    Spearman-correlate our concept weights with external stability rankings.

    Args:
        concept_weights: dict mapping c_idx to weight
        conc2idx: dict mapping concept_name to index
        rankings: dict mapping ranking name to a dict mapping each cleaned concept name to its score in that ranking (output of load_external_rankings)

    Returns:
        results: DataFrame with columns: Ranking, Common concepts, Spearman r, p-value
    """
    idx2conc = {v: k for k, v in conc2idx.items()}

    cleaned_weights = {
        clean_concepts(idx2conc[c]): w
        for c, w in concept_weights.items()
        if w < float("inf") and c in idx2conc
    }

    results = []
    for col in rankings.keys():
        ranked = rankings.get(col, {})
        common = set(cleaned_weights) & set(ranked)
        if len(common) < 3:
            continue
        our = [cleaned_weights[m] for m in common]
        ext = [ranked[m] for m in common]
        rho, p = spearmanr(our, ext)
        results.append(
            {"Ranking": col, "Common concepts": len(common), "Spearman r": round(rho, 4), "p-value": round(p, 10)}
        )

    return pd.DataFrame(results)

def ranking_disagreement(concept_weights, conc2idx, rankings):
    """
    Per-concept rank disagreement between our stability ranking and each
    external ranking, plus the mean disagreement across rankings.

    Our ranking: lower weight/rank = higher variance = more stable.
    External rankings are converted to ranks and, where a higher raw value
    means more stable, reversed so that lower rank = more stable too.
    Disagreement for a concept = |external_rank - our_rank|; NaN if the
    concept is absent from that ranking.

    Args:
        concept_weights: dict mapping c_idx to weight (lower = more stable). Output of compute_concept_weights.
        conc2idx: dict mapping concept_name to index
        rankings: dict mapping ranking name to a dict mapping each cleaned concept name to its raw value (output of load_external_rankings)

    Returns:
        table: DataFrame indexed by concept name with one column per ranking (disagreement, NaN where missing) plus a mean_disagreement column averaged over the rankings that include the concept.
    """
    idx2conc = {v: k for k, v in conc2idx.items()}

    # Our ranking: lower weight = more stable -> rank 1 = most stable.
    our_weights = {
        clean_concepts(idx2conc[c]): w
        for c, w in concept_weights.items()
        if w < float("inf") and c in idx2conc
    }
    our_rank = pd.Series(our_weights).rank(method="min", ascending=True).sort_values()

    # Build a per-concept disagreement table.
    table = pd.DataFrame(index=our_rank.index)
    for col in rankings.keys():
        raw = pd.Series(rankings.get(col, {}))
        # Get subset of concepts present in our ranking
        raw = raw[raw.index.isin(our_rank.index)]
        # ascending=True -> rank 1 = smallest raw value.
        # If higher raw = more stable, reverse so rank 1 = most stable.
        ext_rank = raw.rank(method="min", ascending=True)
        if RANKINGS[col]:
            ext_rank = ext_rank.max() + 1 - ext_rank
        # Only concepts present in both rankings get a value; others -> NaN.
        table[col] = (ext_rank - our_rank).abs()

    table["mean_disagreement"] = table.mean(axis=1, skipna=True).round(2)
    table.index.name = "Concept"
    return table


# CLI

def parse_args():
    p = argparse.ArgumentParser(description="Compute concept stability weights")
    p.add_argument("--config", required=True,
                   help="Path to config JSON (required for concept names and model info)")
    p.add_argument("--epoch", type=int, default=None,
                   help="Checkpoint epoch (default: latest)")
    p.add_argument("--embeddings", default=None,
                   help="Path to pre-computed embeddings .npz "
                        "(from evaluate_pairwise_distances --save_embeddings); "
                        "skips model loading if provided")
    p.add_argument("--rankings", default="data/mapped-and-ranked-lists.tsv",
                   help="Path to mapped-and-ranked-lists.tsv for stability correlations")
    p.add_argument("--output_dir", required=True, help="Directory for output CSVs")
    p.add_argument("--block", type=int, default=256)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = TrainConfig.load_json(args.config)
    device = cfg.get_device()
    conc2idx = cfg.conc2idx
    idx2conc = {v: k for k, v in conc2idx.items()}

    # Load or compute mean embeddings
    if args.embeddings:
        data = np.load(args.embeddings)
        mean_emb = torch.tensor(data["mean_emb"], dtype=torch.float32).to(device)
        valid = torch.tensor(data["valid"], dtype=torch.bool).to(device)
        print(f"Loaded pre-computed embeddings from {args.embeddings}")
    else:
        lang2idx = cfg.lang2idx
        char2idx = cfg.char2idx
        use_fv = cfg.token_type == "featvecs"
        dataset = LexicalDataset(cfg.data, char2idx, conc2idx, lang2idx, to_fv=use_fv)
        dataloader = DataLoader(
            dataset, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=False
        )
        model = build_model(cfg)
        ckpt = load_model_checkpoint(
            device, cfg.model_name, dir=cfg.checkpoint_dir, epoch=args.epoch
        )
        model.load_state_dict(ckpt["model_state_dict"])
        mean_emb, valid = get_mean_embeddings(model, lang2idx, conc2idx, dataloader, use_fv)
        mean_emb = mean_emb.to(device)
        valid = valid.to(device)

    # Compute weights
    weights = compute_concept_weights(mean_emb, valid, block=args.block)

    rows = [
        {
            "Concepticon_Gloss": idx2conc.get(c, f"concept_{c}"),
            "Weight": w,
        }
        for c, w in weights.items()
    ]
    df_weights = pd.DataFrame(rows).sort_values("Weight")
    df_weights["Rank"] = df_weights["Weight"].rank(method="min")
    df_weights.to_csv(out_dir / "concept_weights.csv", index=False)
    print(f"Concept weights saved to {out_dir / 'concept_weights.csv'}")

    # Stability ranking correlations
    if args.rankings:
        rankings = load_rankings(args.rankings)
        df_corr = stability_ranking_correlations(weights, conc2idx, rankings)
        df_corr.to_csv(out_dir / "stability_correlations.csv", index=False)
        print(f"Stability correlations saved to {out_dir / 'stability_correlations.csv'}")
        print(df_corr.to_string(index=False))
        # Per-concept disagreement table
        df_disagree = ranking_disagreement(weights, conc2idx, rankings)
        df_disagree.to_csv(out_dir / "stability_disagreement.csv")
        print(f"Stability disagreement table saved to {out_dir / 'stability_disagreement.csv'}")


if __name__ == "__main__":
    main()
