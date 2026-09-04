"""
Correlate embedding-based language distances with geographic (geodesic) distances.

Reads a pre-computed cosine distance CSV and a languages metadata CSV,
computes haversine distances, and reports the Spearman correlation.

Inputs:
    --distances: path to the cosine distance CSV from evaluate_pairwise_distances; required
    --languages: path to the languages metadata CSV; default=data/all_languages.csv
    --output: path to the output CSV for correlation results; default=results/geo_correlation.csv
    --subsample: optional number of languages to subsample for geo computation; default=None. May run out of memory for large datasets if not set.
    --seed: random seed for subsampling; default=42

Outputs:
    output: CSV file containing the correlation results between embedding distances and geographic distances
    plot: PNG file containing the scatter plot with GAM smooth

Example:
    python -m src.evaluate_geo_correlation \\
        --distances out/cosine_dist.csv \\
        --languages data/all_languages.csv \\
        --output results/geo_correlation.csv \\
        --subsample 500
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from pygam import LinearGAM, s
from matplotlib import pyplot as plt

def compute_geodesic_distances(languages, labels, subsample=None, seed=42):
    """
    Vectorised haversine distances (km) for a list of language glottocodes.

    Args:
        languages: DataFrame with columns glottocode, latitude, longitude
        labels: list of glottocodes (order defines matrix rows/columns)
        subsample: if set, randomly restrict to this many languages
        seed: random seed for subsampling

    Returns:
        geo_dist: (N, N) numpy array in km
        labels: (possibly subsampled) list of glottocodes
    """
    if subsample is not None and len(labels) > subsample:
        rng = np.random.default_rng(seed)
        labels = list(rng.choice(labels, subsample, replace=False))

    coords_df = (
        languages[["glottocode", "latitude", "longitude"]]
        .dropna(subset=["latitude", "longitude"])
        .drop_duplicates(subset=["glottocode"])
    )
    id2coords = {
        row["glottocode"]: (row["latitude"], row["longitude"])
        for _, row in coords_df.iterrows()
    }

    n = len(labels)
    lat = np.full(n, np.nan)
    lon = np.full(n, np.nan)
    has_coords = np.zeros(n, dtype=bool)
    for i, lang in enumerate(labels):
        c = id2coords.get(lang)
        if c is not None:
            lat[i], lon[i] = c
            has_coords[i] = True

    geo = np.zeros((n, n), dtype=float)
    idx = np.where(has_coords)[0]
    if idx.size >= 2:
        lat_r = np.radians(lat[idx])
        lon_r = np.radians(lon[idx])
        dlat = lat_r[:, None] - lat_r[None, :]
        dlon = lon_r[:, None] - lon_r[None, :]
        a = np.clip(
            np.sin(dlat / 2) ** 2
            + np.cos(lat_r[:, None]) * np.cos(lat_r[None, :]) * np.sin(dlon / 2) ** 2,
            0.0,
            1.0,
        )
        geo[np.ix_(idx, idx)] = 6371.0 * 2.0 * np.arcsin(np.sqrt(a))

    return geo, labels


# Plot geographic distance vs embedding distance
def add_gam_smooth(ax, x, y, color='darkred'):
    """
    Fit and plot a GAM smooth (equivalent to geom_smooth(formula=y~s(x), method='gam', se=F, lwd=2))
    """
    x = np.array(x)
    y = np.array(y)

    # Sort by x for smooth line plotting
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]

    # Fit GAM with spline smoother
    gam = LinearGAM(s(0), n_splines=10).fit(x.reshape(-1, 1), y)

    # Predict on a fine grid for smooth curve
    x_grid = np.linspace(x_sorted.min(), x_sorted.max(), 300)
    y_pred = gam.predict(x_grid.reshape(-1, 1))

    ax.plot(x_grid, y_pred, color=color, linewidth=2)


# CLI
def parse_args():
    p = argparse.ArgumentParser(
        description="Correlate embedding language distances with geographic distances"
    )
    p.add_argument("--distances", required=True,
                   help="Cosine distance CSV (from evaluate_pairwise_distances)")
    p.add_argument("--languages", default="data/all_languages.csv",
                   help="Languages metadata CSV with columns: glottocode, latitude, longitude")
    p.add_argument("--output", default="results/geo_correlation.csv", help="Output CSV for correlation results")
    p.add_argument("--subsample", type=int, default=None,
                   help="Subsample N languages for geo computation (default: all)")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for subsampling (default: 42)")    
    return p.parse_args()


def main():
    args = parse_args()

    df_cos = pd.read_csv(args.distances, index_col=0)
    languages = pd.read_csv(args.languages)
    all_labels = list(df_cos.index)

    geo, labels = compute_geodesic_distances(
        languages, all_labels, subsample=args.subsample, seed=args.seed
    )

    # Extract matching submatrix from the cosine distance matrix
    pos = [all_labels.index(l) for l in labels]
    sub_cos = df_cos.values[np.ix_(pos, pos)]

    n = len(labels)
    triu = np.triu_indices(n, k=1)
    geo_vec = geo[triu]
    cos_vec = sub_cos[triu]

    # Exclude pairs where geographic distance is 0 (missing coordinates)
    mask = geo_vec > 0
    geo_vec, cos_vec = geo_vec[mask], cos_vec[mask]

    rho, p = spearmanr(geo_vec, cos_vec)
    print(f"Spearman r (geographic vs. cosine distance): {rho:.4f}  (p = {p:.3e})")
    print(f"Based on {mask.sum():,} language pairs "
          f"({len(labels)} languages, subsample={args.subsample})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        [{"metric": "cosine", "spearman_r": rho, "p_value": p, "n_pairs": int(mask.sum())}]
    ).to_csv(out, index=False)
    print(f"Results saved to {out}")

    # Create plot of geographic vs cosine distances and save it
    plot_path = out.with_suffix(".png")

    plt.figure(figsize=(6, 5))
    ax1 = plt.gca()
    ax1.scatter(geo_vec, cos_vec, s=1, c='teal')
    add_gam_smooth(ax1, geo_vec, cos_vec)
    ax1.set_title(f"Geographic vs Cosine Distance\nSpearman r={rho:.3f}, p={p:.3e}")
    ax1.set_xlabel("Geographic Distance (km)")
    ax1.set_ylabel("Cosine Distance")

    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
