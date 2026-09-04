# DualCWE
Self-Supervised Lexical Representation Learning for Fast, Large-Scale Phylogenetic Inference

This repository implements the DualCWE (dual contrastive word encoder) model.
The commands below reproduce the full pipeline: training, distance computation, tree inference, and evaluation, with downstream analyses applied to a representative seed.
## Replicate study / run experiments
### 0. Setup

Clone/download the repository, navigate to the folder in your terminaml and set up a Python environment (optional) with the required dependencies.

**Create and activate a virtual environment, then install dependencies.**

*Windows (cmd):*
```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

*Linux / macOS*
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**Additional requirements**

- **R** for tree inference.
- **WSL with [QDist](https://birc.au.dk/software/qdist/)** (only on Windows) for computing the Generalized Quartet Distance. On Linux you can run `qdist` directly instead of via WSL, but you'll have to adjust the relevant functions in `src.get_qdist_full.py` and `src.get_qdist_families.py` accordingly.
- **Data files**: the pipeline expects the data files under `data/` (e.g. `lexibank_pruned_wordlist.csv`, `dellert_ranking.csv`, `glottolog.tre`, `all_languages.csv`). The CLTS data is downloaded automatically on first run by `src/ipa2vec.py`.

### 1. Train and evaluate the model

Run the full experiment pipeline (training, pairwise distances, tree inference, and GQD evaluation) across multiple seeds:

```bash
python -m run_experiment --token_type featvecs --model_type dual_contrastive --model_name cwe --out_dir out/dual_contrastive/featvecs/ --results_dir results/dual_contrastive/featvecs/ --checkpoints_dir checkpoints/dual_contrastive/featvecs/
```

The `--token_type` argument selects the segment representation:
- `featvecs` - phonological feature vectors (used in the paper)
- `asjp`     - ASJP codes
- `ipa`      - IPA segments
- `sca`      - SCA sound classes
- `dolgo`    - Dolgopolsky sound classes

To train a single model (without the full pipeline), use `src.train` directly:

```bash
python -m src.train --model_type dual_contrastive --model_name dualcwe --token_type featvecs
```

To evaluate a single tree, use `src.get_gqidst_full` directly. We pass --skip_r to skip tree generation, but need to add a pruned_glottolog_tree.txt to the out_dir (which you can just copy from a different analysis on the same data).
```bash
python -m src.get_qdist_full --skip_r --out_dir out/ldn/ --results results/ldn/full_gqd_results.csv 
```


### 2. Geographic correlation

Correlate embedding-based language distances with geographic (geodesic) distances:

```bash
python -m src.evaluate_geo_correlation --distances out/dual_contrastive/featvecs/seed5/cosine_dist.csv --languages data/all_languages.csv --output results/dual_contrastive/featvecs/seed5/geo_correlation.csv --subsample 500
```

### 3. Per-family evaluation

Compute per-family GQD against the pruned Glottolog tree:

```bash
python -m src.get_qdist_families --dist_csv out/dual_contrastive/featvecs/seed5/cosine_dist.csv --out_dir out/dual_contrastive/featvecs/seed5/families/ --results results/dual_contrastive/featvecs/seed5/family_gqd_results.csv
```

### 4. Concept stability

Compute per-concept stability weights and correlate them with external rankings:

```bash
python -m src.evaluate_concept_weights --config checkpoints/dual_contrastive/featvecs/seed5/cwe_seed5_config.json --embeddings out/dual_contrastive/featvecs/seed5/embeddings.npz --output_dir results/dual_contrastive/featvecs/seed5/
```

### 5. Jäger scripts

The scripts in `src/jäger_scripts/` reproduce the analyses from Jäger (2026) and are, few adjustments to work with my pipeline aside, taken as is from the repo accompanying the study.

**Root and make the tree ultrametric (only for NJ trees)**

If the inferred tree is a NJ tree (not UPGMA), we root and ultrametricize it before plotting:

```bash
python -m src.jäger_scripts.mad out/dual_contrastive/featvecs/seed5/inferred_tree_cosine_dist_nj.txt
Rscript src/jäger_scripts/make_tree_ultrametric.R --rooted_tree out/dual_contrastive/featvecs/seed5/inferred_tree_cosine_dist_nj.txt.rooted --out_dir out/dual_contrastive/featvecs/seed5/
```

**Plot the tree**

```bash
Rscript src/jäger_scripts/plot_tree.R --world_tree out/dual_contrastive/featvecs/seed5/worldtree.ultrametric.nwk --out_dir results/dual_contrastive/featvecs/seed5/
```

**Compare tree to Glottolog**

To get the tree F1-score of family recover, we run:
```bash
python -m src.jäger_scripts.family_recovery --trees out/dual_contrastive/featvecs/seed5/inferred_tree_cosine_dist_nj.txt --out results/dual_contrastive/featvecs/seed5/family_recovery.csv
``` 

## Adding a new model type

This repository currently implements a single model, `DualCWE` (dual contrastive word encoder). The code is structured so that additional model types can be added into the existing pipeline. This document describes the steps involved.

### Overview of the pipeline

The pipeline is split into several modules:

| Module | Purpose |
| ------ | ------- |
| `src/config.py` | `TrainConfig` dataclass and the `build_model()` factory |
| `src/model.py` | Model definitions (currently `DualCWE`) |
| `src/data_prep.py` | Data loading, `LexicalDataset`, `collate_fn`, `LangConcBatchSampler` |
| `src/train.py` | Training loop |
| `src/training_util.py` | Loss functions and checkpointing |
| `src/evaluate_pairwise_distances.py` | Compute pairwise language distance matrices |
| `src/evaluate_concept_weights.py` | Per-concept stability weights |
| `src/evaluate_geo_correlation.py` | Geographic correlation |
| `src/get_qdist_full.py`, `src/get_qdist_families.py` | Tree inference and GQD |
| `run_experiment.py` | Full multi-seed pipeline |

### Steps to add a new model

**1. Define the model class in `src/model.py`**

Add a new `nn.Module` subclass. The model must expose the following interface
so that the rest of the pipeline can use it:

- A `get_representations(x, c=None, l=None)` method that returns a per-(language, concept) embedding tensor of shape `(B, D)`. This is used by `evaluate_pairwise_distances.py` to accumulate mean embeddings. Neither concepts (c), nor languages (l) need to be used by the model - the arguments are just there for compatibility with the evaluation script. 
- A `representation_dim` attribute giving the embedding dimension `D`. This is read by   `evaluate_pairwise_distances.py`.

**2. Register the model in `src/config.py`**

Add a function that initializes your model and adjust the `build_model()` function:

```python
def build_model(cfg: TrainConfig) -> nn.Module:
    if cfg.model_type == "dual_contrastive":
        return build_dual_cwe(cfg)
    elif cfg.model_type == "my_new_model":
        return build_my_new_model(cfg)
    else:
        raise ValueError(f"Unsupported model type: {cfg.model_type}")
```

If the new model needs hyperparameters that are not already in `TrainConfig`, add them as new fields to the dataclass.

**3. Wire up the training loop in `src/train.py`**

The training loop in `src/train.py` currently assumes the dual-contrastive objective (two forward passes with stochastic augmentations, then an NT-Xent loss). If your model uses a different objective, add a new branch in the training loop. You may also need to adjust the dataloader setup if your model does not use the `LangConcBatchSampler`. You'll find placeholders for both the dataset/-loader setup as well as the validation loop.
If you use a loss function different than the NT-Xent-loss, you may want to add it to `src/training_utils.py` and import and implement it in `src/train.py`.

**4. Add command-line arguments**

Add any new hyperparameters to `parse_args()` in `src/train.py`. If you run full experiments via `run_experiment.py`, also add the corresponding arguments there and forward them to `src.train`.

**5. Evaluation scripts**

The evaluation scripts (`evaluate_pairwise_distances.py`, `evaluate_concept_weights.py`) call `model.get_representations()` and read `model.representation_dim`. As long as your model implements these, no changes are needed. If your model produces embeddings differently (for example, a single language-level embedding rather than per-concept embeddings), you will need to adjust `get_mean_embeddings()` in `evaluate_pairwise_distances.py`.
