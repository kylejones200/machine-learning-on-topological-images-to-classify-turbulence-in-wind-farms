#!/usr/bin/env python3
"""
Power Curve Degradation Detection Using Persistent Laplacians
Tracks gradual performance degradation using spectral signatures.
"""

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ripser import ripser
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        import yaml as _yaml

        return _yaml.safe_load(_f) or {}


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

np.random.seed(42)


def fetch_nrel_wind_data(lat=41.5, lon=-100.5, years=None):
    if years is None:
        years = [2010, 2011, 2012]
    """Simulate NREL Wind Toolkit data fetch."""
    logger.info(f"Simulating NREL wind data fetch for location ({lat}, {lon})")

    n_records = 365 * 24 * 12 * len(years)
    timestamps = pd.date_range(
        start=f"{years[0]}-01-01", periods=n_records, freq="5min"
    )

    hours = np.array([t.hour + t.minute / 60 for t in timestamps])
    days = np.array([t.dayofyear for t in timestamps])

    seasonal = 2 * np.sin(2 * np.pi * days / 365)
    diurnal = 1.5 * np.sin(2 * np.pi * hours / 24)

    windspeed_80m = 8.5 + seasonal + diurnal + np.random.normal(0, 2, n_records)
    windspeed_80m = np.clip(windspeed_80m, 0, 25)

    wind_direction = (
        180 + 60 * np.sin(2 * np.pi * days / 365) + np.random.normal(0, 15, n_records)
    )
    wind_direction = wind_direction % 360

    temperature = (
        15 + 10 * np.cos(2 * np.pi * days / 365) + np.random.normal(0, 3, n_records)
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "windspeed_80m": windspeed_80m,
            "wind_direction": wind_direction,
            "temperature": temperature,
        }
    )

    logger.info(f"Fetched {len(df)} records spanning {len(years)} years")
    return df


def simulate_turbine_power_degraded(windspeed, degradation_level=0, rated_power=2.0):
    """
    Simulate turbine power with degradation.
    degradation_level: 0 (healthy), 1 (2% degraded), 2 (5% degraded), 3 (10% degraded)
    """
    cut_in = 3.0
    rated_speed = 12.0
    cut_out = 25.0

    degradation_factors = [1.0, 0.98, 0.95, 0.90]
    degradation_factor = degradation_factors[degradation_level]

    power = np.zeros_like(windspeed)

    for i, ws in enumerate(windspeed):
        if ws < cut_in or ws > cut_out:
            power[i] = 0
        elif ws < rated_speed:
            power[i] = rated_power * ((ws - cut_in) / (rated_speed - cut_in)) ** 3
        else:
            power[i] = rated_power

        # Apply degradation
        power[i] *= degradation_factor

        # Add noise (more noise with degradation)
        noise_level = 0.03 * (1 + degradation_level * 0.1)
        power[i] += np.random.normal(0, noise_level * rated_power)
        power[i] = max(0, power[i])

    return power


def create_degradation_scenarios(df, n_windows=120, window_size=288):
    """
    Create labeled windows with different degradation levels.
    Label: 0=healthy, 1=degraded (2%+)
    """
    logger.info(f"\nCreating {n_windows} labeled windows (healthy vs degraded)...")

    windows = []
    labels = []

    max_start = len(df) - window_size
    starts = np.random.choice(max_start, n_windows, replace=False)

    for idx, start in enumerate(starts):
        window_df = df.iloc[start : start + window_size].copy()

        # Distribute degradation levels
        degradation_level = np.where(
            idx < n_windows // 2, 0, np.random.choice([1, 2, 3])
        )

        power = simulate_turbine_power_degraded(
            window_df["windspeed_80m"].values, degradation_level=degradation_level
        )

        window_df["power"] = power
        window_df["degradation_level"] = degradation_level

        pd.concat([windows, window_df])
        labels.append(1 if degradation_level > 0 else 0)

    logger.info(
        f"Created {sum(labels)} degraded windows and {len(labels) - sum(labels)} healthy windows"
    )
    return windows, np.array(labels)


def compute_graph_laplacian(points, k=8):
    """Compute graph Laplacian from k-NN graph."""
    n = len(points)

    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    distances, indices = nbrs.kneighbors(points)

    # Build adjacency matrix (Gaussian kernel)
    row_idx = []
    col_idx = []
    weights = []

    for i in range(n):
        for j_idx in range(1, k + 1):  # Skip self (index 0)
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            weight = np.exp(-(dist**2) / (2 * 0.5**2))  # sigma = 0.5
            pd.concat([row_idx, i])
            pd.concat([col_idx, j])
            pd.concat([weights, weight])

    # Symmetric adjacency
    row_idx_sym = row_idx + col_idx
    col_idx_sym = col_idx + row_idx
    weights_sym = weights + weights

    W = csr_matrix((weights_sym, (row_idx_sym, col_idx_sym)), shape=(n, n))

    # Degree matrix
    D = np.array(W.sum(axis=1)).flatten()
    csr_matrix((D, (range(n), range(n))), shape=(n, n))

    # Normalized Laplacian: L = I - D^{-1/2} W D^{-1/2}
    D_inv_sqrt = np.sqrt(1.0 / (D + 1e-10))
    D_inv_sqrt_mat = csr_matrix((D_inv_sqrt, (range(n), range(n))), shape=(n, n))

    L = csr_matrix(np.eye(n)) - D_inv_sqrt_mat @ W @ D_inv_sqrt_mat

    return L


def compute_laplacian_features(window_df, n_eigenvalues=10):
    """
    Compute persistent Laplacian features.
    Spectral signature changes with degradation.
    """
    power = window_df["power"].values
    windspeed = window_df["windspeed_80m"].values

    # Normalize
    power_norm = (power - power.min()) / (power.max() - power.min() + 1e-8)
    wind_norm = (windspeed - windspeed.min()) / (
        windspeed.max() - windspeed.min() + 1e-8
    )

    # Subsample to reduce computation
    n_samples = min(200, len(power_norm))
    indices = np.random.choice(len(power_norm), n_samples, replace=False)

    points = np.column_stack([power_norm[indices], wind_norm[indices]])

    # Compute Laplacian
    L = compute_graph_laplacian(points, k=8)

    # Compute eigenvalues
    try:
        n_eigs = min(n_eigenvalues, L.shape[0] - 2)
        eigenvalues, _ = eigsh(L, k=n_eigs, which="SM")
        eigenvalues = np.sort(eigenvalues)
    except Exception:
        eigenvalues = np.zeros(n_eigenvalues)

    features = {}

    # Spectral features
    for i in range(min(n_eigenvalues, len(eigenvalues))):
        features[f"eigenvalue_{i}"] = eigenvalues[i] if i < len(eigenvalues) else 0

    if len(eigenvalues) > 0:
        features["spectral_gap"] = (
            eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
        )
        features["eigenvalue_mean"] = np.mean(eigenvalues)
        features["eigenvalue_std"] = np.std(eigenvalues)
        features["eigenvalue_sum"] = np.sum(eigenvalues)
    else:
        features["spectral_gap"] = 0
        features["eigenvalue_mean"] = 0
        features["eigenvalue_std"] = 0
        features["eigenvalue_sum"] = 0

    # Add persistence features
    result = ripser(points, maxdim=1)
    diagrams = result["dgms"]

    for dim in [0, 1]:
        dgm = diagrams[dim]
        finite_dgm = dgm[dgm[:, 1] != np.inf]
        if len(finite_dgm) > 0:
            lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
            features[f"H{dim}_max_lifetime"] = np.max(lifetimes)
            features[f"H{dim}_sum_lifetime"] = np.sum(lifetimes)
        else:
            features[f"H{dim}_max_lifetime"] = 0
            features[f"H{dim}_sum_lifetime"] = 0

    # Statistical features
    features["power_mean"] = power.mean()
    features["power_std"] = power.std()
    features["wind_mean"] = windspeed.mean()
    features["power_cv"] = power.std() / (power.mean() + 1e-8)

    return features


def extract_all_features(windows, labels):
    """Extract Laplacian and persistence features."""
    logger.info("\nExtracting Laplacian spectral features...")

    feature_list = []
    for i, window_df in enumerate(windows):
        if i % 20 == 0:
            logger.info(f"  Processing window {i + 1}/{len(windows)}")

        features = compute_laplacian_features(window_df, n_eigenvalues=10)
        pd.concat([feature_list, features])

    X = pd.DataFrame(feature_list)
    y = labels

    logger.info(f"\nFeature matrix: {X.shape}")
    logger.info(f"Label distribution: Degraded={sum(y)}, Healthy={len(y) - sum(y)}")

    return X, y


def train_and_evaluate_models(X, y):
    """Train and evaluate classifiers."""
    logger.info("=== TRAINING AND EVALUATION ===")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    logger.info(f"\nTrain set: {len(X_train)} samples")
    logger.info(f"Test set: {len(X_test)} samples")

    models = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM (Linear)": SVC(kernel="linear", random_state=42, probability=True),
        "SVM (RBF)": SVC(kernel="rbf", random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    results = {}

    for name, model in models.items():
        logger.info(f"\n{name}:")
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = (
            model.predict_proba(X_test)[:, 1]
            if hasattr(model, "predict_proba")
            else None
        )

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

        logger.info(f"  Accuracy: {acc:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")
        if auc is not None:
            logger.info(f"  AUC: {auc:.3f}")

        results[name] = {
            "model": model,
            "accuracy": acc,
            "f1": f1,
            "auc": auc,
            "y_test": y_test,
            "y_pred": y_pred,
        }

    return results


def generate_visualizations(
    windows, labels, X, y, results, out_dir, plot: bool = False
):
    """Generate comprehensive visualizations."""
    logger.info("=== GENERATING VISUALIZATIONS ===")

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Model comparison
    logger.info("\n1. Model comparison...")
    if plot:
        fig, axes = plt.subplots(
            1, 2, figsize=tuple(config.get("output", {}).get("figsize", [12, 4]))
        )

        model_names = list(results.keys())
        accuracies = [results[m]["accuracy"] for m in model_names]
        f1s = [results[m]["f1"] for m in model_names]

        axes[0].bar(range(len(model_names)), accuracies, color="#2b2b2b", alpha=0.85)
        axes[0].set_xticks(range(len(model_names)))
        axes[0].set_xticklabels(model_names, rotation=45, ha="right")
        axes[0].set_ylabel("Accuracy")
        axes[0].set_title("Degradation Detection Accuracy")
        axes[0].set_ylim([0, 1])
        axes[0].spines["top"].set_visible(False)
        axes[0].spines["right"].set_visible(False)

        axes[1].bar(range(len(model_names)), f1s, color="#d62728", alpha=0.85)
        axes[1].set_xticks(range(len(model_names)))
        axes[1].set_xticklabels(model_names, rotation=45, ha="right")
        axes[1].set_ylabel("F1 Score")
        axes[1].set_title("Degradation Detection F1 Score")
        axes[1].set_ylim([0, 1])
        axes[1].spines["top"].set_visible(False)
        axes[1].spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(out_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("  Saved: model_comparison.png")

        # 2. Eigenvalue spectra comparison
        logger.info("2. Eigenvalue spectra comparison...")
        healthy_idx = np.where(labels == 0)[0][0]
        degraded_idx = np.where(labels == 1)[0][0]

        healthy_eigs = [X.iloc[healthy_idx][f"eigenvalue_{i}"] for i in range(10)]
        degraded_eigs = [X.iloc[degraded_idx][f"eigenvalue_{i}"] for i in range(10)]

        plt.figure(figsize=(10, 6))
        plt.plot(
            range(10),
            healthy_eigs,
            "o-",
            label="Healthy",
            color="#696969",
            linewidth=2,
            markersize=8,
        )
        plt.plot(
            range(10),
            degraded_eigs,
            "s-",
            label="Degraded",
            color="#d62728",
            linewidth=2,
            markersize=8,
        )
        plt.xlabel("Eigenvalue Index")
        plt.ylabel("Eigenvalue")
        plt.title("Laplacian Eigenvalue Spectrum: Healthy vs Degraded")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(out_dir / "eigenvalue_spectra.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("  Saved: eigenvalue_spectra.png")

        # 3. Spectral gap distribution
        logger.info("3. Spectral gap distribution...")
        healthy_gaps = X[y == 0]["spectral_gap"]
        degraded_gaps = X[y == 1]["spectral_gap"]

        plt.figure(figsize=(10, 6))
        plt.hist(
            healthy_gaps,
            bins=20,
            alpha=0.6,
            label="Healthy",
            color="#696969",
            edgecolor="#2b2b2b",
        )
        plt.hist(
            degraded_gaps,
            bins=20,
            alpha=0.6,
            label="Degraded",
            color="#d62728",
            edgecolor="#2b2b2b",
        )
        plt.xlabel("Spectral Gap")
        plt.ylabel("Count")
        plt.title("Spectral Gap Distribution: Healthy vs Degraded")
        plt.legend(frameon=False)
        plt.tight_layout()
        plt.savefig(
            out_dir / "spectral_gap_distribution.png", dpi=300, bbox_inches="tight"
        )
        plt.close()
        logger.info("  Saved: spectral_gap_distribution.png")

        # 4. Feature importance
        logger.info("4. Feature importance...")
        if "Random Forest" in results:
            rf_model = results["Random Forest"]["model"]
            importances = rf_model.feature_importances_
            indices = np.argsort(importances)[::-1][:10]

            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(
                range(len(indices)), importances[indices], color="#2b2b2b", alpha=0.85
            )
            ax.set_xticks(range(len(indices)))
            ax.set_xticklabels([X.columns[i] for i in indices], rotation=45, ha="right")
            ax.set_ylabel("Importance")
            ax.set_title("Top 10 Feature Importances (Random Forest)")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            plt.tight_layout()
            plt.savefig(
                out_dir / "feature_importance.png", dpi=300, bbox_inches="tight"
            )
            plt.close()
        logger.info("  Saved: feature_importance.png")

    logger.info("\nAll visualizations generated successfully!")


def main():
    """Main execution."""
    logger.info("POWER CURVE DEGRADATION USING PERSISTENT LAPLACIANS")

    df = fetch_nrel_wind_data()
    windows, labels = create_degradation_scenarios(df, n_windows=120, window_size=288)
    X, y = extract_all_features(windows, labels)
    results = train_and_evaluate_models(X, y)

    out_dir = Path(__file__).parent / "figures_degradation"
    generate_visualizations(windows, labels, X, y, results, out_dir)

    logger.info("=== FINAL SUMMARY ===")
    best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_result = results[best_model_name]
    logger.info(f"\nBest Model: {best_model_name}")
    logger.info(f"  Accuracy: {best_result['accuracy']:.3f}")
    logger.info(f"  F1 Score: {best_result['f1']:.3f}")

    logger.info(f"\nVisualizations saved to: {out_dir}/")
    logger.info("\nAnalysis complete!")


if __name__ == "__main__":
    main()
