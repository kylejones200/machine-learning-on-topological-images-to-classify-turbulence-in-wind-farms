"""
Turbulence Intensity Classification Using Persistence Images and CNNs
Classifies high vs low turbulence from SCADA using topological deep learning
"""

import bisect
import logging
import os
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

_REGIME_CUTS = [3, 12]


def _turbine_state(w: float, rated_power: float) -> tuple[float, float, float]:
    """Return (target_power, target_rpm, target_pitch) for wind speed w."""
    match bisect.bisect_right(_REGIME_CUTS, w):
        case 0:
            return 0.0, 0.0, 90.0
        case 1:
            return (
                rated_power * ((w - 3) / 9) ** 2.5,
                10 + (w - 3) * 5,
                5 + (12 - w) * 2,
            )
        case _:
            return rated_power, 55 + (w - 12) * 0.2, 2.0


def load_config(config_path=None):
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = Path(__file__).parent / "config.yaml"
    if not config_path.exists():
        return {}
    with open(config_path) as _f:
        return _yaml.safe_load(_f) or {}


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
config = load_config()
# For persistence
import matplotlib.pyplot as plt

# For deep learning
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from ripser import ripser
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

# Configuration
NREL_API_KEY = os.getenv("NREL_API_KEY", "")
NREL_API_URL = (
    "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv"
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def fetch_nrel_wind_data(config=None):
    """Fetch wind data from NREL."""
    if config is None:
        config = {}
    nrel = config.get("nrel", {})
    lat = nrel.get("lat", 41.5)
    lon = nrel.get("lon", -93.5)
    years = nrel.get("years", [2017])
    attributes = nrel.get("attributes", "windspeed_100m,temperature_100m")
    interval = nrel.get("interval", "60")
    email = os.getenv("NREL_EMAIL", "")
    all_data = []

    for year in years:
        logger.info(f"   Fetching year {year}...")

        params = {
            "api_key": NREL_API_KEY,
            "wkt": f"POINT({lon} {lat})",
            "attributes": attributes,
            "names": str(year),
            "utc": "true",
            "leap_day": "false",
            "interval": interval,
            "email": email,
        }

        try:
            response = requests.get(NREL_API_URL, params=params, timeout=120)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.warning("Year %s fetch failed: %s", year, e)
            continue

        lines = response.text.strip().split("\n")
        data_start = 0
        for i, line in enumerate(lines):
            if line.startswith("Year,"):
                data_start = i + 1
                break

        data_text = "\n".join(lines[data_start:])
        df_year = pd.read_csv(
            StringIO(data_text),
            header=None,
            names=[
                "Year",
                "Month",
                "Day",
                "Hour",
                "Minute",
                "windspeed_100m",
                "temperature_100m",
            ],
        )

        df_year["time"] = pd.to_datetime(
            df_year[["Year", "Month", "Day", "Hour", "Minute"]]
        )
        pd.concat([all_data, df_year])
        logger.info(f"     ✓ Fetched {len(df_year):,} records")

    if not all_data:
        return None

    return pd.concat(all_data, ignore_index=True).sort_values("time")


def simulate_turbulence_and_turbine(wind_df, rated_power=2000):
    """
    Simulate turbulence intensity and turbine response.

    TI varies with time of day (stability):
    - Day (unstable): TI = 0.18-0.22
    - Night (stable): TI = 0.06-0.09
    - Neutral: TI = 0.10-0.14
    """
    df = wind_df.copy()
    n = len(df)

    # Determine atmospheric stability from hour
    hour = df["time"].dt.hour.values

    # Simple stability model
    # Day (6-18): unstable, high TI
    # Night (18-6): stable, low TI
    # Transition periods: neutral
    ti = np.zeros(n)
    for i in range(n):
        h = hour[i]
        if 8 <= h < 16:  # Daytime, unstable
            ti[i] = np.random.uniform(0.16, 0.22)
        elif h >= 20 or h < 4:  # Nighttime, stable
            ti[i] = np.random.uniform(0.05, 0.09)
        else:  # Transition, neutral
            ti[i] = np.random.uniform(0.10, 0.14)

    # Add turbulent fluctuations to wind speed
    wind_mean = df["windspeed_100m"].values
    wind_turbulent = np.zeros(n)

    for i in range(n):
        # Add multi-scale turbulence
        # Large scale (slow)
        large_scale = np.where(
            i == 0,
            0,
            0.95 * wind_turbulent[i - 1]
            + 0.05 * np.random.randn() * wind_mean[i] * ti[i],
        )

        # Small scale (fast)
        small_scale = np.random.randn() * wind_mean[i] * ti[i] * 0.3

        wind_turbulent[i] = wind_mean[i] + large_scale + small_scale
        wind_turbulent[i] = max(0, wind_turbulent[i])

    # Simulate turbine response
    rotor_speed = np.zeros(n)
    power = np.zeros(n)
    pitch = np.zeros(n)

    for i in range(1, n):
        w = wind_turbulent[i]

        # Power curve
        target_power, target_rpm, target_pitch = _turbine_state(w, rated_power)

        # Dynamics with lag
        rotor_speed[i] = 0.85 * rotor_speed[i - 1] + 0.15 * target_rpm
        power[i] = 0.75 * power[i - 1] + 0.25 * target_power
        pitch[i] = 0.90 * pitch[i - 1] + 0.10 * target_pitch

        # Add noise
        rotor_speed[i] += np.random.randn() * 0.5
        power[i] += np.random.randn() * 20
        pitch[i] += np.random.randn() * 0.5

        # Clip
        rotor_speed[i] = np.clip(rotor_speed[i], 0, 70)
        power[i] = np.clip(power[i], 0, rated_power * 1.1)
        pitch[i] = np.clip(pitch[i], 0, 90)

    df["turbulence_intensity"] = ti
    df["wind_turbulent"] = wind_turbulent
    df["rotor_speed"] = rotor_speed
    df["power"] = power
    df["pitch"] = pitch

    return df


def diagram_to_image(diagram, resolution=20, sigma=0.1):
    """
    Convert persistence diagram to persistence image.

    Args:
        diagram: Persistence diagram (nx2 array)
        resolution: Image resolution (pixels per side)
        sigma: Gaussian smoothing parameter

    Returns:
        Image array (resolution x resolution)
    """
    # Filter finite points
    finite_mask = np.isfinite(diagram[:, 1])
    finite_pts = diagram[finite_mask]

    if len(finite_pts) == 0:
        return np.zeros((resolution, resolution))

    # Compute persistence
    births = finite_pts[:, 0]
    deaths = finite_pts[:, 1]
    persistences = deaths - births

    # Set up image grid
    b_min, b_max = max(0, births.min() - 0.1), births.max() + 0.1
    d_min, d_max = b_min, deaths.max() + 0.1

    x = np.linspace(b_min, b_max, resolution)
    y = np.linspace(d_min, d_max, resolution)
    X, Y = np.meshgrid(x, y)

    # Create image by summing Gaussians weighted by persistence
    image = np.zeros((resolution, resolution))

    for i in range(len(finite_pts)):
        b, d, p = births[i], deaths[i], persistences[i]

        # Gaussian centered at (b, d)
        gaussian = np.exp(-((X - b) ** 2 + (Y - d) ** 2) / (2 * sigma**2))
        image += p * gaussian

    # Normalize
    if image.max() > 0:
        image = image / image.max()

    return image


def create_persistence_image_dataset(df, window_size=10, resolution=20):
    """
    Create dataset of persistence images with turbulence labels.

    Args:
        df: DataFrame with turbine data
        window_size: Window size in samples
        resolution: Image resolution

    Returns:
        images, labels
    """
    logger.info("\n   Creating persistence images...")

    images = []
    labels = []

    n = len(df)
    for start in range(0, n - window_size + 1, window_size):
        end = start + window_size
        window = df.iloc[start:end]

        # Extract 4D trajectory
        wind = window["wind_turbulent"].values
        rotor = window["rotor_speed"].values
        power = window["power"].values
        pitch = window["pitch"].values

        X = np.column_stack([wind, rotor, power, pitch])
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-10)

        # Compute persistence
        try:
            result = ripser(X, maxdim=1)
            diagrams = result["dgms"]

            # Convert H0 and H1 to images
            img_h0 = diagram_to_image(diagrams[0], resolution=resolution)
            img_h1 = diagram_to_image(
                diagrams[1] if len(diagrams) > 1 else np.empty((0, 2)),
                resolution=resolution,
            )

            # Stack as 2-channel image
            img = np.stack([img_h0, img_h1], axis=0)

            # Label: mean TI in window
            ti_mean = window["turbulence_intensity"].mean()
            if ti_mean < 0.10:
                label = 0  # Low turbulence
            elif ti_mean > 0.15:
                label = 1  # High turbulence
            else:
                continue  # Skip moderate (ambiguous)

            pd.concat([images, img])
            pd.concat([labels, label])

        except Exception:
            continue

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    logger.info(f"     Created {len(images)} persistence images")
    logger.info(f"     Low turbulence: {(labels == 0).sum()}")
    logger.info(f"     High turbulence: {(labels == 1).sum()}")

    return images, labels


class PersistenceImageDataset(Dataset):
    """PyTorch dataset for persistence images."""

    def __init__(self, images, labels):
        self.images = torch.FloatTensor(images)
        self.labels = torch.LongTensor(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class PersistenceCNN(nn.Module):
    """CNN for persistence image classification."""

    def __init__(self, input_channels=2, num_classes=2):
        super().__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)

        # Fully connected layers
        # After 2 pooling layers: 20x20 -> 10x10 -> 5x5
        self.fc1 = nn.Linear(32 * 5 * 5, 64)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # Conv blocks
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))

        # Flatten
        x = x.view(-1, 32 * 5 * 5)

        # FC layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x


def train_model(model, train_loader, val_loader, num_epochs=30, learning_rate=0.001):
    """Train the CNN."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    best_val_acc = 0

    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()

        train_acc = 100.0 * train_correct / train_total

        # Validation
        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(DEVICE), labels.to(DEVICE)
                outputs = model(images)
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()

        val_acc = 100.0 * val_correct / val_total

        if val_acc > best_val_acc:
            best_val_acc = val_acc

        if (epoch + 1) % 5 == 0:
            logger.info(
                f"   Epoch {epoch + 1}/{num_epochs}: "
                f"Train Acc = {train_acc:.2f}%, Val Acc = {val_acc:.2f}%"
            )

        scheduler.step()

    return model


def evaluate_model(model, test_loader):
    """Evaluate the trained model."""
    model.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            outputs = model(images)
            probs = F.softmax(outputs, dim=1)

            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    acc = accuracy_score(all_labels, all_preds)
    auc = roc_auc_score(all_labels, all_probs)

    return acc, auc, all_preds, all_probs, all_labels


def main(plot: bool = False):
    np.random.seed(config.get("data", {}).get("seed", 42))
    torch.manual_seed(42)

    logger.info("Turbulence Classification Using Persistence Images & CNN")

    # 1. Fetch wind data
    logger.info("\n1. Fetching NREL wind data...")
    wind_data = fetch_nrel_wind_data(config=config)
    if wind_data is None:
        logger.error("Failed to fetch data", exc_info=True)
        return
    logger.info(f"   Total records: {len(wind_data):,}")

    # 2. Simulate turbulence and turbine
    logger.info("\n2. Simulating turbulence and turbine response...")
    df = simulate_turbulence_and_turbine(wind_data)

    ti_low = (df["turbulence_intensity"] < 0.10).sum()
    ti_high = (df["turbulence_intensity"] > 0.15).sum()
    logger.info(f"   Low TI (<0.10): {ti_low} ({ti_low / len(df) * 100:.1f}%)")
    logger.info(f"   High TI (>0.15): {ti_high} ({ti_high / len(df) * 100:.1f}%)")

    # 3. Create persistence image dataset
    logger.info("\n3. Creating persistence image dataset...")
    images, labels = create_persistence_image_dataset(df, window_size=10, resolution=20)

    # 4. Split data
    logger.info("\n4. Splitting data...")
    # Chronological split
    split_idx = int(0.7 * len(images))
    X_train, X_test = images[:split_idx], images[split_idx:]
    y_train, y_test = labels[:split_idx], labels[split_idx:]

    # Further split train into train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    logger.info(f"   Train: {len(X_train)} samples")
    logger.info(f"   Val: {len(X_val)} samples")
    logger.info(f"   Test: {len(X_test)} samples")

    # Create data loaders
    train_dataset = PersistenceImageDataset(X_train, y_train)
    val_dataset = PersistenceImageDataset(X_val, y_val)
    test_dataset = PersistenceImageDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # 5. Train model
    logger.info("\n5. Training CNN...")
    model = PersistenceCNN(input_channels=2, num_classes=2).to(DEVICE)
    logger.info(f"   Using device: {DEVICE}")

    model = train_model(
        model, train_loader, val_loader, num_epochs=30, learning_rate=0.001
    )

    # 6. Evaluate
    logger.info("\n6. Evaluating on test set...")
    acc, auc, preds, probs, labels_true = evaluate_model(model, test_loader)

    logger.info(f"\n   Test Accuracy: {acc * 100:.2f}%")
    logger.info(f"   Test AUC: {auc:.3f}")
    logger.info(
        f"\n{classification_report(labels_true, preds, target_names=['Low TI', 'High TI'])}"
    )

    # 7. Visualizations
    logger.info("\n7. Generating visualizations...")
    out_dir = Path("figures_turbulence")
    out_dir.mkdir(exist_ok=True)

    # ROC curve
    from sklearn.metrics import roc_curve

    fpr, tpr, _ = roc_curve(labels_true, probs)

    if plot:
        fig, ax = plt.subplots(
            figsize=tuple(config.get("output", {}).get("figsize", [8, 8]))
        )
        ax.plot(fpr, tpr, "k-", linewidth=2, label=f"AUC = {auc:.3f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, label="Random")
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)
        ax.set_title(
            "ROC Curve: Turbulence Classification", fontsize=12, fontweight="normal"
        )
        ax.legend(frameon=False, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        plt.tight_layout()
        plt.savefig(out_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
        plt.close()

    logger.info(f"   Saved visualizations to {out_dir}/")

    logger.info("=== TURBULENCE CLASSIFICATION COMPLETE ===")
    logger.info(f"\nCNN on persistence images: {acc * 100:.1f}% accuracy")
    logger.info("No specialized sensors required - SCADA only")
    logger.info("Enables:")
    logger.info("  - Turbulence-aware load monitoring")
    logger.info("  - Adaptive control strategies")
    logger.info("  - Site assessment validation")


if __name__ == "__main__":
    main()
