"""
Train a model on lexibank_pruned_wordlist.csv.

To add a new model type or training objective, adjust parse_args(), the
TrainConfig construction, the dataloader setup, and the training loop below.
You may also need to update src/config.py (TrainConfig dataclass and the
build_model() factory) and run_experiment.py.

Inputs:
    --forms_path: path to lexibank_pruned_wordlist.csv; default=data/lexibank_pruned_wordlist.csv
    --concepts_path: path to dellert_ranking.csv; default=data/dellert_ranking.csv
    --token_type: segment representation to use (ipa, sca, dolgo, asjp, featvecs); default=featvecs
    --to_rank: keep only the top-N ranked concepts; default=all
    --model_type: model architecture; default=dual_contrastive
    --embedding_dim: embedding dimension; default=256
    --encoder_layers: number of encoder layers; default=1
    --encoder_dropout: encoder dropout probability; default=0.1
    --feat_dim: feature vector dimension (only used when token_type=featvecs); default=39
    --num_epochs: number of training epochs; default=5
    --batch_size: batch size; default=512
    --learning_rate: learning rate; default=1e-3
    --val_ratio: validation split ratio; default=0.01
    --seed: random seed; default=1
    --temperature: temperature for contrastive loss; default=0.3
    --swap_p: probability of swapping two characters in a word; default=0.0
    --dup_p: probability of duplicating a character in a word; default=0.1
    --noise_p: probability of adding noise to a character in a word; default=0.5
    --del_p: probability of deleting a character in a word; default=0.0
    --input_dropout: dropout probability for input tokens; default=0.1
    --num_langs_per_batch: number of languages per batch for dual contrastive learning; default=32
    --num_concepts_per_lang: number of concepts per language for dual contrastive learning; default=30
    --lang_loss_weight: weight for the language-level contrastive loss; default=0.2
    --model_name: name prefix for checkpoint files and config JSON; default=dualcwe
    --checkpoint_dir: directory for checkpoints and config; default=checkpoints
    --load_model: resume training from the latest available checkpoint

Outputs:
    config: <model_name>_config.json in --checkpoint_dir
    checkpoints: <model_name>_epoch*.pth in --checkpoint_dir

Example:
    python -m src.train \
        --model_type dual_contrastive \
        --model_name dualcwe \
        --token_type asjp \
        --num_epochs 10
"""

import argparse
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm.auto import trange
from einops import rearrange
import random
import numpy as np

from src.config import TrainConfig, build_model
from src.data_prep import LexicalDataset, data_preprocessing, collate_fn, LangConcBatchSampler
from src.training_util import (
    load_model_checkpoint,
    save_model_checkpoint,
    train_val_test_split,
    nt_xent_loss
)


def parse_args():
    p = argparse.ArgumentParser(description="Train a model on lexibank_pruned_wordlist.csv")

    # Data paths
    p.add_argument("--forms_path", default="data/lexibank_pruned_wordlist.csv",
                   help="Path to lexibank_pruned_wordlist.csv")
    p.add_argument("--concepts_path", default="data/dellert_ranking.csv",
                   help="Path to dellert_ranking.csv")
    p.add_argument("--token_type", default="featvecs",
                   choices=["ipa", "sca", "dolgo", "asjp", "featvecs"],
                   help="Segment representation to use (default: featvecs)")
    p.add_argument("--to_rank", type=int, default=None,
                   help="Keep only the top-N ranked concepts. Default: use all.")

    # Architecture
    p.add_argument("--model_type", default="dual_contrastive")
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--encoder_layers", type=int, default=1)
    p.add_argument("--encoder_dropout", type=float, default=0.1)
    p.add_argument("--feat_dim", type=int, default=39,
                   help="Feature vector dimension (only used when token_type=featvecs)")

    # Training
    p.add_argument("--num_epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--learning_rate", type=float, default=1e-3)
    p.add_argument("--val_ratio", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=1)

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
                   help="Weight for the language-level contrastive loss in dual contrastive learning")

    # Bookkeeping
    p.add_argument("--model_name", default="dualcwe",
                   help="Name prefix for checkpoint files and config JSON")
    p.add_argument("--checkpoint_dir", default="checkpoints")
    p.add_argument("--load_model", action="store_true",
                   help="Resume training from the latest available checkpoint")

    return p.parse_args()



def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main():
    args = parse_args()
    set_seeds(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


    # Load raw data
    forms = pd.read_csv(args.forms_path, encoding="utf-8")
    ranked_concepts = pd.read_csv(args.concepts_path)

    data, vocab, langs, concs, langs2fam = data_preprocessing(
        forms,
        ranked_concepts=ranked_concepts,
        to_rank=args.to_rank,
        token_type=args.token_type,
    )

    print(f"Languages : {len(langs)}")
    print(f"Concepts  : {len(concs)}")
    print(f"Data points: {len(data)}")

    # Build vocabularies / index mappings
    special_tokens = ["[PAD]", "[CLS]", "^", "$", "[MASK]"]
    vocab = special_tokens + vocab          # pad=0, cls=1, bos=2, eos=3, mask=4
    char2idx = {ch: i for i, ch in enumerate(vocab)}
    lang2idx = {lang: i for i, lang in enumerate(langs)}
    conc2idx = {conc: i for i, conc in enumerate(concs)}

    use_fv = args.token_type == "featvecs"

    # Datasets / dataloaders

    if args.model_type == "dual_contrastive": # No val data
        train_dataset = LexicalDataset(data, char2idx, conc2idx, lang2idx, to_fv=use_fv)
        train_loader = DataLoader(
            train_dataset, collate_fn=collate_fn, batch_sampler=LangConcBatchSampler(train_dataset, N=args.num_langs_per_batch, K=args.num_concepts_per_lang)
        )
        print(f"Vocab size: {len(vocab)}")
        print(f"Train / val: {len(data)} / 0")
    else:
        raise NotImplementedError(f"DataLoader for model type {args.model_type} not implemented yet. Check train.py and add a new branch for your model type.")
        # Example for a standard DataLoader (not dual contrastive):
        # train_data, val_data, _ = train_val_test_split(
        #         data, random_seed=args.seed, test_ratio=0.0, val_ratio=args.val_ratio
        # )
        # train_dataset = LexicalDataset(train_data, char2idx, conc2idx, lang2idx, to_fv=use_fv)
        # val_dataset = LexicalDataset(val_data, char2idx, conc2idx, lang2idx, to_fv=use_fv)
        # train_loader = DataLoader(
        #     train_dataset, collate_fn=collate_fn, shuffle=True, batch_size=args.batch_size)
        #         )
        # val_loader = DataLoader(
        #     val_dataset, batch_size=args.batch_size, collate_fn=collate_fn
        # )

        # print(f"Vocab size: {len(vocab)}")
        # print(f"Train / val: {len(train_data)} / {len(val_data)}")

    # Config (save before training so it can be used by eval scripts)
    cfg = TrainConfig(
        model_type=args.model_type,
        embedding_dim=args.embedding_dim,
        encoder_layers=args.encoder_layers,
        encoder_dropout=args.encoder_dropout,
        feat_dim=args.feat_dim,
        learning_rate=args.learning_rate,
        temperature=args.temperature,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        seed=args.seed,
        val_ratio=args.val_ratio,
        token_type=args.token_type,
        to_rank=args.to_rank,
        model_name=args.model_name,
        checkpoint_dir=args.checkpoint_dir,
        load_model=args.load_model,
        num_langs=len(langs),
        num_concepts=len(concs),
        vocab_size=max(char2idx.values()) + 1,
        lang2idx=lang2idx,
        conc2idx=conc2idx,
        char2idx=char2idx,
        langs2fam=langs2fam,
        data=data,
        device=str(device),
        swap_p=args.swap_p,
        dup_p=args.dup_p,
        noise_p=args.noise_p,
        del_p=args.del_p,
        input_dropout=args.input_dropout,
        num_langs_per_batch=args.num_langs_per_batch,
        num_concepts_per_lang=args.num_concepts_per_lang
    )
    config_path = Path(args.checkpoint_dir) / f"{args.model_name}_config.json"
    cfg.save_json(config_path)
    print(f"Config saved to {config_path}")


    # Model
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    if args.load_model:
        checkpoint = load_model_checkpoint(device, args.model_name, dir=args.checkpoint_dir)
        if checkpoint is not None:
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            print(f"Resumed from epoch {checkpoint['epoch']}")
    else: # clear the checkpoint directory if not resuming; just a safety measure because I had some weird interactions
        pth = Path(args.checkpoint_dir)
        pth.mkdir(parents=True, exist_ok=True)
        for f in pth.glob(f"{args.model_name}_epoch*.pth"):
            f.unlink()

    # Training loop
    best_val_loss = float("inf")
    best_epoch = -1
    t = trange(args.num_epochs)

    for epoch in t:
        # Train
        model.train()
        total_loss = 0.0
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            concept_tokens = batch["concept_tokens"].to(device)
            lang_tokens = batch["lang_tokens"].to(device)
            targets = (
                batch["char_featvecs"].to(device) if use_fv else batch["chars"].to(device)
            )
            if args.model_type == "dual_contrastive":
                word_out1, lang_out1 = model(targets, c=concept_tokens, l=lang_tokens)
                word_out2, lang_out2 = model(targets, c=concept_tokens, l=lang_tokens)
                # Rearrange such that positive pairs are adjacent
                word_outputs = rearrange([word_out1, word_out2], 'b c d -> (c b) d')
                word_loss = nt_xent_loss(word_outputs, temperature=args.temperature)
                if args.lang_loss_weight > 0.0:
                    lang_outputs = rearrange([lang_out1, lang_out2], 'b c d -> (c b) d')
                    lang_loss = nt_xent_loss(lang_outputs, temperature=args.temperature)
                    loss = word_loss + args.lang_loss_weight * lang_loss
                else:
                    loss = word_loss
            else:
                raise ValueError(f"Unknown model type: {args.model_type}")
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            t.set_description(f"Epoch {epoch + 1}, batch {i + 1}/{len(train_loader)}")

        avg_train = total_loss / len(train_loader)

        # Validate
        # This is a placeholder to run your models through a validation set. You may need to implement a validation dataloader and loss function for your specific model type.
        if args.model_type == "dual_contrastive":
            val_loader = []  # No validation for dual contrastive learning
        else:
            raise NotImplementedError(f"Validation for model type {args.model_type} not implemented yet. Check train.py and add a new branch for your model type.")
        model.eval()
        val_loss_sum = 0.0
        with torch.no_grad():
            for batch in val_loader:
                concept_tokens = batch["concept_tokens"].to(device)
                lang_tokens = batch["lang_tokens"].to(device)
                targets = (
                    batch["char_featvecs"].to(device) if use_fv else batch["chars"].to(device)
                )

                # Example for a standard validation loop (not dual contrastive):
                # outputs = model(targets, c=concept_tokens, l=lang_tokens)
                # loss = your_loss_function(outputs, targets)  # Replace with your actual loss function

                val_loss_sum += loss.item()

        avg_val = val_loss_sum / len(val_loader) if len(val_loader) > 0 else 0
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_epoch = epoch + 1

        # Save checkpoint at the end of each epoch
        save_model_checkpoint(
            model, optimizer, epoch, avg_val,
            fn=args.model_name, dir=args.checkpoint_dir,
        )
        t.set_postfix({
            "train": f"{avg_train:.4f}",
            "val": f"{avg_val:.4f}",
            "best": f"{best_val_loss:.4f}",
            "best_epoch": f"{best_epoch}"
        })

    t.close()
    print(f"Training complete. Final train loss: {avg_train:.4f}. Best validation loss: {best_val_loss:.4f}.")


if __name__ == "__main__":
    main()
