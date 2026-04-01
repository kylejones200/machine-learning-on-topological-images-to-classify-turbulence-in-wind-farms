#!/usr/bin/env python3
"""
Power Curve Degradation Detection Using Persistent Laplacians
Tracks gradual performance degradation using spectral signatures.
"""

import logging
import os
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from io import StringIO
from pathlib import Path

from ripser import ripser
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh

import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

np.random.seed(42)

# NREL API configuration (real data)
NREL_API_URL = (
    "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv"
)
DEFAULT_NREL_EMAIL = "kyletjones@gmail.com"


def _get_nrel_api_key() -> str:
    """Return the NREL API key from the environment."""
    api_key = os.environ.get("NREL_API_KEY")
    if not api_key:
        raise RuntimeError(
            "NREL_API_KEY environment variable is not set. "
            "Export your key, e.g. `export NREL_API_KEY='your-key-here'`."
        )
    return api_key


def _normalize_nrel_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize NREL Wind Toolkit column names to the ones used in this script.
    """
    cols = {c.lower(): c for c in df.columns}

    # Timestamp
    ts_col = None
    for key in ['timestamp', 'time', 'datetime', 'date_time']:
        if key in cols:
            ts_col = cols[key]
            break
    if ts_col is None:
        raise ValueError(f"Could not find a timestamp column in NREL data. Columns: {list(df.columns)}")

    # Wind speed
    ws_col = None
    for key in ['wind_speed', 'windspeed', 'windspeed_80m', 'wind_speed_80m']:
        if key in cols:
            ws_col = cols[key]
            break
    if ws_col is None:
        raise ValueError("Could not find a wind speed column in NREL data.")

    # Wind direction
    wd_col = None
    for key in ['wind_direction', 'winddir', 'wind_direction_80m']:
        if key in cols:
            wd_col = cols[key]
            break
    if wd_col is None:
        raise ValueError("Could not find a wind direction column in NREL data.")

    # Temperature
    temp_col = None
    for key in ['air_temperature', 'temperature', 'temperature_80m']:
        if key in cols:
            temp_col = cols[key]
            break
    if temp_col is None:
        raise ValueError("Could not find a temperature column in NREL data.")

    df = df.rename(
        columns={
            ts_col: 'timestamp',
            ws_col: 'windspeed_80m',
            wd_col: 'wind_direction',
            temp_col: 'temperature',
        }
    )
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df[['timestamp', 'windspeed_80m', 'wind_direction', 'temperature']]


def fetch_nrel_wind_data(
    lat: float = 41.5,
    lon: float = -100.5,
    years: List[int] | None = None,
    email: str = DEFAULT_NREL_EMAIL,
) -> pd.DataFrame:
    """Fetch real wind data from the NREL Wind Toolkit API."""
    if years is None:
        years = [2010, 2011, 2012]

    api_key = _get_nrel_api_key()
    logger.info(
        "Requesting NREL Wind Toolkit data lat=%.3f lon=%.3f years=%s", lat, lon, years
    )

    all_frames: List[pd.DataFrame] = []
    for year in years:
        params: Dict[str, Any] = {
            "api_key": api_key,
            "lat": lat,
            "lon": lon,
            "year": year,
            "interval": 5,
            "email": email,
        }
        try:
            response = requests.get(NREL_API_URL, params=params, timeout=60)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("NREL request failed for year=%s: %s", year, exc)
            raise RuntimeError(
                f"NREL API request failed for year {year}. See logs for details."
            ) from exc

        year_df = pd.read_csv(StringIO(response.text))
        year_df = _normalize_nrel_columns(year_df)
        all_frames.append(year_df)

    df = (
        pd.concat(all_frames, axis=0)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )
    logger.info("Fetched %d NREL records spanning %d year(s)", len(df), len(years))
    return df

DEGRADATION_CUT_IN_SPEED_MPS = 3.0
DEGRADATION_RATED_SPEED_MPS = 12.0
DEGRADATION_CUT_OUT_SPEED_MPS = 25.0
DEGRADATION_RATED_POWER_MW = 2.0
DEGRADATION_FACTORS = [1.0, 0.98, 0.95, 0.90]


def simulate_turbine_power_degraded(
    windspeed: np.ndarray, degradation_level: int = 0, rated_power: float = DEGRADATION_RATED_POWER_MW
) -> np.ndarray:
    """Simulate turbine power with degradation."""
    power = np.zeros_like(windspeed)

    degradation_factor = DEGRADATION_FACTORS[degradation_level]

    for i, ws in enumerate(windspeed):
        if ws < DEGRADATION_CUT_IN_SPEED_MPS or ws > DEGRADATION_CUT_OUT_SPEED_MPS:
            power[i] = 0.0
        elif ws < DEGRADATION_RATED_SPEED_MPS:
            power[i] = rated_power * (
                (ws - DEGRADATION_CUT_IN_SPEED_MPS)
                / (DEGRADATION_RATED_SPEED_MPS - DEGRADATION_CUT_IN_SPEED_MPS)
            ) ** 3
        else:
            power[i] = rated_power

        power[i] *= degradation_factor

        noise_level = 0.03 * (1 + degradation_level * 0.1)
        power[i] += np.random.normal(0, noise_level * rated_power)
        power[i] = max(0.0, power[i])

    return power


def create_degradation_scenarios(
    df: pd.DataFrame, n_windows: int = 120, window_size: int = 288
) -> Tuple[List[pd.DataFrame], np.ndarray]:
    """Create labeled windows with different degradation levels."""
    logger.info("Creating %d labeled windows (healthy vs degraded)", n_windows)
    
    windows = []
    labels = []
    
    max_start = len(df) - window_size
    starts = np.random.choice(max_start, n_windows, replace=False)
    
    for idx, start in enumerate(starts):
        window_df = df.iloc[start:start+window_size].copy()
        
        # Distribute degradation levels
        if idx < n_windows // 2:
            degradation_level = 0  # Healthy
        else:
            degradation_level = np.random.choice([1, 2, 3])  # Degraded
        
        power = simulate_turbine_power_degraded(
            window_df['windspeed_80m'].values,
            degradation_level=degradation_level
        )
        
        window_df['power'] = power
        window_df['degradation_level'] = degradation_level
        
        windows.append(window_df)
        labels.append(1 if degradation_level > 0 else 0)
    
    logger.info(
        "Created %d degraded windows and %d healthy windows",
        int(sum(labels)),
        int(len(labels) - sum(labels)),
    )
    return windows, np.array(labels)

def compute_graph_laplacian(points, k=8):
    """Compute graph Laplacian from k-NN graph."""
    n = len(points)
    
    # Build k-NN graph
    nbrs = NearestNeighbors(n_neighbors=k+1).fit(points)
    distances, indices = nbrs.kneighbors(points)
    
    # Build adjacency matrix (Gaussian kernel)
    row_idx = []
    col_idx = []
    weights = []
    
    for i in range(n):
        for j_idx in range(1, k+1):  # Skip self (index 0)
            j = indices[i, j_idx]
            dist = distances[i, j_idx]
            weight = np.exp(-dist**2 / (2 * 0.5**2))  # sigma = 0.5
            row_idx.append(i)
            col_idx.append(j)
            weights.append(weight)
    
    # Symmetric adjacency
    row_idx_sym = row_idx + col_idx
    col_idx_sym = col_idx + row_idx
    weights_sym = weights + weights
    
    W = csr_matrix((weights_sym, (row_idx_sym, col_idx_sym)), shape=(n, n))
    
    # Degree matrix
    D = np.array(W.sum(axis=1)).flatten()
    D_mat = csr_matrix((D, (range(n), range(n))), shape=(n, n))
    
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
    power = window_df['power'].values
    windspeed = window_df['windspeed_80m'].values
    
    # Normalize
    power_norm = (power - power.min()) / (power.max() - power.min() + 1e-8)
    wind_norm = (windspeed - windspeed.min()) / (windspeed.max() - windspeed.min() + 1e-8)
    
    # Subsample to reduce computation
    n_samples = min(200, len(power_norm))
    indices = np.random.choice(len(power_norm), n_samples, replace=False)
    
    points = np.column_stack([power_norm[indices], wind_norm[indices]])
    
    # Compute Laplacian
    L = compute_graph_laplacian(points, k=8)
    
    # Compute eigenvalues
    try:
        n_eigs = min(n_eigenvalues, L.shape[0] - 2)
        eigenvalues, _ = eigsh(L, k=n_eigs, which='SM')
        eigenvalues = np.sort(eigenvalues)
    except:
        eigenvalues = np.zeros(n_eigenvalues)
    
    features = {}
    
    # Spectral features
    for i in range(min(n_eigenvalues, len(eigenvalues))):
        features[f'eigenvalue_{i}'] = eigenvalues[i] if i < len(eigenvalues) else 0
    
    if len(eigenvalues) > 0:
        features['spectral_gap'] = eigenvalues[1] - eigenvalues[0] if len(eigenvalues) > 1 else 0
        features['eigenvalue_mean'] = np.mean(eigenvalues)
        features['eigenvalue_std'] = np.std(eigenvalues)
        features['eigenvalue_sum'] = np.sum(eigenvalues)
    else:
        features['spectral_gap'] = 0
        features['eigenvalue_mean'] = 0
        features['eigenvalue_std'] = 0
        features['eigenvalue_sum'] = 0
    
    # Add persistence features
    result = ripser(points, maxdim=1)
    diagrams = result['dgms']
    
    for dim in [0, 1]:
        dgm = diagrams[dim]
        finite_dgm = dgm[dgm[:, 1] != np.inf]
        if len(finite_dgm) > 0:
            lifetimes = finite_dgm[:, 1] - finite_dgm[:, 0]
            features[f'H{dim}_max_lifetime'] = np.max(lifetimes)
            features[f'H{dim}_sum_lifetime'] = np.sum(lifetimes)
        else:
            features[f'H{dim}_max_lifetime'] = 0
            features[f'H{dim}_sum_lifetime'] = 0
    
    # Statistical features
    features['power_mean'] = power.mean()
    features['power_std'] = power.std()
    features['wind_mean'] = windspeed.mean()
    features['power_cv'] = power.std() / (power.mean() + 1e-8)
    
    return features

def extract_all_features(
    windows: List[pd.DataFrame], labels: np.ndarray
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Extract Laplacian and persistence features."""
    logger.info("Extracting Laplacian spectral features")

    feature_list: List[Dict[str, float]] = []
    for i, window_df in enumerate(windows):
        if i % 20 == 0:
            logger.info("Processing window %d/%d", i + 1, len(windows))

        features = compute_laplacian_features(window_df, n_eigenvalues=10)
        feature_list.append(features)

    X = pd.DataFrame(feature_list)
    y = labels

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    logger.info("Feature matrix shape: %s", X.shape)
    logger.info(
        "Label distribution: degraded=%d healthy=%d",
        int(sum(y)),
        int(len(y) - sum(y)),
    )

    return X, y


def train_and_evaluate_models(
    X: pd.DataFrame, y: np.ndarray
) -> Dict[str, Dict[str, Any]]:
    """Train and evaluate classifiers."""
    logger.info("=" * 60)
    logger.info("TRAINING AND EVALUATION")
    logger.info("=" * 60)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info("Train set: %d samples", len(X_train))
    logger.info("Test set: %d samples", len(X_test))
    
    models: Dict[str, Any] = {
        "Logistic Regression": LogisticRegression(random_state=42, max_iter=1000),
        "SVM (Linear)": SVC(kernel="linear", random_state=42, probability=True),
        "SVM (RBF)": SVC(kernel="rbf", random_state=42, probability=True),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, random_state=42
        ),
    }

    results: Dict[str, Dict[str, Any]] = {}

    for name, model in models.items():
        logger.info("Training model: %s", name)
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

        logger.info("  Accuracy=%.3f F1=%.3f AUC=%s", acc, f1, f"{auc:.3f}" if auc else "n/a")

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
    windows: List[pd.DataFrame],
    labels: np.ndarray,
    X: pd.DataFrame,
    y: np.ndarray,
    results: Dict[str, Dict[str, Any]],
    out_dir: Path | str,
) -> None:
    """Generate comprehensive visualizations."""
    logger.info("=" * 60)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 60)

    out_dir = Path(out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)
    
    # 1. Model comparison
    logger.info("Creating model comparison plots")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    model_names = list(results.keys())
    accuracies = [results[m]['accuracy'] for m in model_names]
    f1s = [results[m]['f1'] for m in model_names]
    
    axes[0].bar(range(len(model_names)), accuracies, color='#2b2b2b', alpha=0.85)
    axes[0].set_xticks(range(len(model_names)))
    axes[0].set_xticklabels(model_names, rotation=45, ha='right')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Degradation Detection Accuracy')
    axes[0].set_ylim([0, 1])
    axes[0].spines['top'].set_visible(False)
    axes[0].spines['right'].set_visible(False)
    
    axes[1].bar(range(len(model_names)), f1s, color='#d62728', alpha=0.85)
    axes[1].set_xticks(range(len(model_names)))
    axes[1].set_xticklabels(model_names, rotation=45, ha='right')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('Degradation Detection F1 Score')
    axes[1].set_ylim([0, 1])
    axes[1].spines['top'].set_visible(False)
    axes[1].spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(out_dir / "model_comparison.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved model comparison to %s", out_dir / "model_comparison.png")
    
    # 2. Eigenvalue spectra comparison
    logger.info("Creating eigenvalue spectra comparison")
    healthy_idx = np.where(labels == 0)[0][0]
    degraded_idx = np.where(labels == 1)[0][0]
    
    healthy_eigs = [X.iloc[healthy_idx][f'eigenvalue_{i}'] for i in range(10)]
    degraded_eigs = [X.iloc[degraded_idx][f'eigenvalue_{i}'] for i in range(10)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(range(10), healthy_eigs, 'o-', label='Healthy', color='#696969', linewidth=2, markersize=8)
    plt.plot(range(10), degraded_eigs, 's-', label='Degraded', color='#d62728', linewidth=2, markersize=8)
    plt.xlabel('Eigenvalue Index')
    plt.ylabel('Eigenvalue')
    plt.title('Laplacian Eigenvalue Spectrum: Healthy vs Degraded')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "eigenvalue_spectra.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info("Saved eigenvalue spectra to %s", out_dir / "eigenvalue_spectra.png")
    
    # 3. Spectral gap distribution
    logger.info("Creating spectral gap distribution plot")
    healthy_gaps = X[y == 0]['spectral_gap']
    degraded_gaps = X[y == 1]['spectral_gap']
    
    plt.figure(figsize=(10, 6))
    plt.hist(healthy_gaps, bins=20, alpha=0.6, label='Healthy', color='#696969', edgecolor='#2b2b2b')
    plt.hist(degraded_gaps, bins=20, alpha=0.6, label='Degraded', color='#d62728', edgecolor='#2b2b2b')
    plt.xlabel('Spectral Gap')
    plt.ylabel('Count')
    plt.title('Spectral Gap Distribution: Healthy vs Degraded')
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_dir / "spectral_gap_distribution.png", dpi=300, bbox_inches="tight")
    plt.close()
    logger.info(
        "Saved spectral gap distribution to %s", out_dir / "spectral_gap_distribution.png"
    )
    
    # 4. Feature importance
    logger.info("Creating feature importance plot for Random Forest")
    if 'Random Forest' in results:
        rf_model = results['Random Forest']['model']
        importances = rf_model.feature_importances_
        indices = np.argsort(importances)[::-1][:10]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(indices)), importances[indices], color='#2b2b2b', alpha=0.85)
        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([X.columns[i] for i in indices], rotation=45, ha='right')
        ax.set_ylabel('Importance')
        ax.set_title('Top 10 Feature Importances (Random Forest)')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig(out_dir / "feature_importance.png", dpi=300, bbox_inches="tight")
        plt.close()
        logger.info("Saved feature importance to %s", out_dir / "feature_importance.png")

    logger.info("All visualizations generated successfully")

def main() -> None:
    """Main execution."""
    logger.info("=" * 60)
    logger.info("POWER CURVE DEGRADATION USING PERSISTENT LAPLACIANS")
    logger.info("=" * 60)

    df = fetch_nrel_wind_data()
    windows, labels = create_degradation_scenarios(df, n_windows=120, window_size=288)
    X, y = extract_all_features(windows, labels)
    results = train_and_evaluate_models(X, y)

    out_dir = Path(__file__).parent / "figures_degradation"
    generate_visualizations(windows, labels, X, y, results, out_dir)

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    best_model_name = max(results.keys(), key=lambda k: results[k]["accuracy"])
    best_result = results[best_model_name]
    logger.info(
        "Best model=%s accuracy=%.3f f1=%.3f",
        best_model_name,
        best_result["accuracy"],
        best_result["f1"],
    )
    logger.info("Visualizations saved to %s", out_dir)
    logger.info("Analysis complete")

if __name__ == "__main__":
    main()

