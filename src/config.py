"""
Configuration dataclass and model factory.

To add a new model type:
    1. Add a new model class in src/model.py.
    2. Add a new branch to the build_model() function below.
    3. Add any new hyperparameters to the TrainConfig dataclass.
    4. Wire up the new model in src/train.py and run_experiment.py.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from src.model import DualCWE


@dataclass
class TrainConfig:
    # Architecture
    embedding_dim: int = 256
    encoder_layers: int = 1
    encoder_dropout: float = 0.1
    feat_dim: int = 39  # feature-vector dimension (relevant when token_type="featvecs"; adjust only if you use different feature vectors)
    model_type: str = "dual_contrastive"

    # Training
    learning_rate: float = 1e-3
    batch_size: int = 512 # unused for "dual_contrastive" model type, which uses a custom batch sampler
    num_epochs: int = 15
    seed: int = 1
    val_ratio: float = 0.01 # unused for "dual_contrastive" model type

    # Contrastive learning
    temperature: float = 0.3  # temperature for contrastive loss
    swap_p: float = 0.0  # probability of swapping two characters in a word
    dup_p: float = 0.1  # probability of duplicating a character in a word
    noise_p: float = 0.5  # probability of adding noise to a character in a word
    del_p: float = 0.0  # probability of deleting a character in a word
    input_dropout: float = 0.1  # dropout probability for input tokens
    num_langs_per_batch: int = 32  # languages per batch for dual contrastive learning
    num_concepts_per_lang: int = 30  # concepts per language for dual contrastive learning

    # Data
    token_type: str = "featvecs"  # "ipa" | "sca" | "dolgo" | "asjp" | "featvecs"
    to_rank: Optional[int] = None  # keep top-N ranked concepts; None = all; set to N=105 to use the sample of Jäger (2026)

    # Bookkeeping
    model_name: str = "dualcwe"
    checkpoint_dir: str = "checkpoints"
    load_model: bool = False

    # Data dimensions (populated after data loading)
    num_langs: Optional[int] = None
    num_concepts: Optional[int] = None
    vocab_size: Optional[int] = None

    # Mappings and data (saved to JSON for evaluation scripts)
    lang2idx: Optional[Dict[str, int]] = None
    conc2idx: Optional[Dict[str, int]] = None
    char2idx: Optional[Dict[str, int]] = None
    langs2fam: Optional[Dict[str, str]] = None
    data: Optional[List] = None  # list of [lang, concept, form, family]

    # Device (stored as string, e.g. "cuda" or "cpu")
    device: Optional[str] = None

    def get_device(self) -> torch.device:
        if self.device:
            return torch.device(self.device)
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def save_json(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2, sort_keys=True, ensure_ascii=False)

    @classmethod
    def load_json(cls, path: Path) -> "TrainConfig":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


# Weight initialisation
def _init_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Embedding):
        nn.init.xavier_uniform_(module.weight)


# Model factory
def build_dual_cwe(cfg: TrainConfig) -> DualCWE:
    """Instantiate a DualCWE model from a TrainConfig and move it to the configured device."""
    if None in (cfg.num_concepts, cfg.vocab_size):
        raise ValueError(
            "num_concepts, num_langs, and vocab_size must be set before building the model."
        )
    model = DualCWE(
        vocab_size=cfg.vocab_size,
        num_concepts=cfg.num_concepts,
        num_languages=cfg.num_langs,
        embedding_dim=cfg.embedding_dim,
        encoder_dropout=cfg.encoder_dropout,
        input_dropout=cfg.input_dropout,
        num_layers=cfg.encoder_layers,
        feat_dim=cfg.feat_dim,
        swap_p=cfg.swap_p,
        dup_p=cfg.dup_p,
        noise_p=cfg.noise_p,
        del_p=cfg.del_p,
        langs_per_batch=cfg.num_langs_per_batch,
        concepts_per_lang=cfg.num_concepts_per_lang

    )
    model.apply(_init_weights)
    return model.to(cfg.get_device())

def build_model(cfg: TrainConfig) -> nn.Module:
    """Instantiate a model based on the model_type in the TrainConfig."""
    if cfg.model_type == "dual_contrastive":
        return build_dual_cwe(cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")