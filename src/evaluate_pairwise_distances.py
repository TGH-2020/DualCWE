"""
Compute pairwise cosine language-distance matrix from model embeddings.

For each language pair (i, j), the distance is the mean cosine distance over
all concepts where both languages have a valid form.

Inputs:
    --config: path to the training configuration JSON; default=checkpoints/dualcwe_config.json
    --eval_epoch: checkpoint epoch to load; default=latest
    --output: path to the output cosine distance CSV; required
    --save_embeddings: optional path to save mean embeddings as .npz for reuse
    --block: block size for blockwise distance computation; default=256

Outputs:
    output: symmetric cosine distance CSV (languages x languages)
    save_embeddings: optional .npz file with pre-computed mean embeddings for reuse

Example:
    python -m src.evaluate_pairwise_distances \\
        --config checkpoints/dualcwe_config.json \\
        --output out/cosine_dist.csv \\
        --save_embeddings out/embeddings.npz
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import TrainConfig, build_model
from src.data_prep import LexicalDataset, collate_fn
from src.training_util import load_model_checkpoint


# Embedding accumulation
def get_mean_embeddings(model, lang2idx, conc2idx, dataloader, use_fv=False):
    """
    Accumulate per-(language, concept) mean representations.

    Args:
        model: trained model
        lang2idx: dict mapping glottocode to language index
        conc2idx: dict mapping concept name to concept index
        dataloader: DataLoader over the lexical dataset
        use_fv: if True, use phonological feature vectors as input

    Returns:
        mean_emb: (L, C, D) tensor on the model's device
        valid: (L, C) bool mask indicating cells with at least one example
    """
    device = next(model.parameters()).device
    model.eval()

    L = len(lang2idx)
    C = len(conc2idx)
    D = model.representation_dim

    sum_emb = torch.zeros(L, C, D, device=device)
    count_emb = torch.zeros(L, C, device=device)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Accumulating embeddings"):
            lang_tokens = batch["lang_tokens"].to(device)
            concept_tokens = batch["concept_tokens"].to(device)
            x = batch["char_featvecs"].to(device) if use_fv else batch["chars"].to(device)
            mu = model.get_representations(x, c=concept_tokens, l=lang_tokens)
            for i in range(lang_tokens.size(0)):
                l = lang_tokens[i].item()
                c = concept_tokens[i].item()
                sum_emb[l, c] += mu[i]
                count_emb[l, c] += 1

    mean_emb = sum_emb / count_emb.clamp_min(1).unsqueeze(-1)
    valid = count_emb > 0

    return mean_emb, valid


# Distance computation
def pairwise_cosine_distances(mean_emb, valid, idx2lang, block=256):
    """
    Blockwise pairwise cosine distance matrix. Blocks of size `block` are used to reduce memory usage and allow computation on large datasets.

    Distance(i, j) = mean over concepts c where both languages have valid
    embeddings of (1 - cosine_similarity(emb_i_c, emb_j_c)).

    Args:
        mean_emb: (L, C, D) tensor of mean embeddings
        valid: (L, C) bool mask of which language-concept pairs exist
        idx2lang: dict mapping language index to glottocode
        block: block size for blockwise computation

    Returns:
        cosine_dist: (L, L) numpy array
        labels: list of language glottocodes in row/column order
    """
    device = mean_emb.device
    L = mean_emb.shape[0]
    normed = mean_emb / mean_emb.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    cosine_dist = torch.zeros(L, L, device=device)

    for i0 in range(0, L, block):
        i1 = min(i0 + block, L)
        A = normed[i0:i1]
        Av = valid[i0:i1]
        for j0 in range(0, L, block):
            j1 = min(j0 + block, L)
            B = normed[j0:j1]
            Bv = valid[j0:j1]
            vm = Av.unsqueeze(1) & Bv.unsqueeze(0)          # (B1, B2, C)
            sim = torch.einsum("acd,bcd->abc", A, B) # (B1, B2, C)
            dist = 1.0 - sim
            cosine_dist[i0:i1, j0:j1] = (
                (dist * vm).sum(2) / vm.sum(2).clamp_min(1e-6)
            )

    labels = [idx2lang[i] for i in range(L) if i in idx2lang]
    return cosine_dist.cpu().numpy(), labels



# CLI
def parse_args():
    p = argparse.ArgumentParser(description="Compute pairwise cosine language distances")
    p.add_argument("--config", default="checkpoints/dualcwe_config.json", help="Path to config JSON")
    p.add_argument("--eval_epoch", type=int, default=None,
                   help="Checkpoint epoch to load (default: latest)")
    p.add_argument("--output", required=True, help="Output cosine distance CSV path")
    p.add_argument("--save_embeddings", default=None,
                   help="Optional path to save mean embeddings as .npz for reuse")
    p.add_argument("--block", type=int, default=256,
                   help="Block size for blockwise distance computation")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = TrainConfig.load_json(args.config)
    device = cfg.get_device()

    lang2idx = cfg.lang2idx
    conc2idx = cfg.conc2idx
    char2idx = cfg.char2idx
    idx2lang = {v: k for k, v in lang2idx.items()}
    use_fv = cfg.token_type == "featvecs"

    dataset = LexicalDataset(cfg.data, char2idx, conc2idx, lang2idx, to_fv=use_fv)
    dataloader = DataLoader(
        dataset, batch_size=cfg.batch_size, collate_fn=collate_fn, shuffle=False
    )

    model = build_model(cfg)
    
    checkpoint = load_model_checkpoint(
        device, cfg.model_name, dir=cfg.checkpoint_dir, epoch=args.eval_epoch
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    mean_emb, valid = get_mean_embeddings(model, lang2idx, conc2idx, dataloader, use_fv)
    cosine_dist, labels = pairwise_cosine_distances(mean_emb, valid, idx2lang, block=args.block)


    if args.save_embeddings:
        emb_path = Path(args.save_embeddings)
        emb_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            emb_path,
            mean_emb=mean_emb.cpu().numpy(),
            valid=valid.cpu().numpy(),
        )
        print(f"Embeddings saved to {emb_path}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cosine_dist, index=labels, columns=labels).to_csv(out)
    print(f"Cosine distance matrix saved to {out}")


if __name__ == "__main__":
    main()
