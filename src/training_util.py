"""Loss functions, checkpointing utilities, and data-split helpers."""

import random
from pathlib import Path

import torch
import torch.nn.functional as F


# Checkpointing
def save_model_checkpoint(
    model,
    optimizer,
    epoch: int,
    val_loss: float,
    dir: str = "checkpoints",
    fn: str = "model_checkpoint",
) -> None:
    pth = Path(dir)
    pth.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "val_loss": val_loss,
        },
        pth / f"{fn}_epoch{epoch}.pth",
    )


def load_model_checkpoint(device, fn: str, dir: str = "checkpoints", epoch=None):
    """
    Load a model checkpoint.

    Args:
        device: target device for torch.load
        fn:     model name (filename prefix, without epoch suffix)
        dir:    checkpoint directory
        epoch:  specific epoch to load; if None, loads the latest available
    Returns:
        checkpoint dict or None if not found
    """
    pth = Path(dir)
    if epoch is not None:
        ckpt_path = pth / f"{fn}_epoch{epoch}.pth"
    else:
        candidates = sorted(
            pth.glob(f"{fn}_epoch*.pth"),
            key=lambda x: int(x.stem.split("epoch")[-1]),
        )
        if not candidates:
            print(f"No checkpoints found for '{fn}' in '{dir}'")
            return None
        ckpt_path = candidates[-1]

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}")
        return None

    return torch.load(ckpt_path, map_location=device)


# Loss functions
def nt_xent_loss(z, temperature=0.075):
    """
    Normalised temperature-scaled cross-entropy (NT-Xent) loss.

    Args:
        z: [2N, D] embeddings where positive pairs are adjacent
           (indices 0,1), (2,3), ..., (2N-2, 2N-1)
        temperature: temperature scalar
    Returns:
        scalar loss
    """
    device = z.device
    N = z.size(0) // 2  # number of original words

    # Normalize the embeddings
    z = F.normalize(z, dim=1)
    # Cosine similarity matrix
    sim_matrix = torch.matmul(z, z.T)  # [2N, 2N]

    # Remove self-similarity
    mask = torch.eye(2 * N, dtype=torch.bool, device=device)
    sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

    # Divide by temperature
    sim_matrix = sim_matrix / temperature

    # Positive indices: (0,1), (2,3), ..., (2N-2, 2N-1)
    pos_indices = torch.arange(0, 2 * N, 2, device=device)
    positives = sim_matrix[pos_indices, pos_indices + 1]
    positives = torch.cat([positives, sim_matrix[pos_indices + 1, pos_indices]])

    # For each anchor, compute denominator over all other items
    sim_exp = torch.exp(sim_matrix)
    denom = sim_exp.sum(dim=1)
    denom_pos = torch.cat([
        denom[pos_indices],
        denom[pos_indices + 1]
    ])  # shape: [2N]

    # Numerator is exp(sim(positive1, positive2))
    pos_sims = torch.exp(positives)

    # Compute the loss for both directions
    loss = -torch.log(pos_sims / denom_pos.flatten())
    return loss.mean()

# Data splits
def train_val_test_split(data, test_ratio=0.15, val_ratio=0.15, random_seed=42):
    """
    Split data into train / validation / test sets.

    Args:
        data:        list of data points
        test_ratio:  fraction of data for the test set
        val_ratio:   fraction of data for the validation set
        random_seed: reproducibility seed
    Returns:
        train_data, val_data, test_data
    """
    random.seed(random_seed)
    data_shuffled = data.copy()
    random.shuffle(data_shuffled)
    n = len(data_shuffled)
    test_size = int(n * test_ratio)
    val_size = int(n * val_ratio)
    train_data = data_shuffled[: n - test_size - val_size]
    val_data = data_shuffled[n - test_size - val_size : n - test_size]
    test_data = data_shuffled[n - test_size :]
    return train_data, val_data, test_data

