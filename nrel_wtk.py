"""NREL Wind Toolkit CSV fetch via config.yaml and environment secrets."""

from __future__ import annotations

import logging
import os
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import requests
import yaml

try:
    from dotenv import load_dotenv
except ImportError:

    def load_dotenv(*_args, **_kwargs) -> bool:
        return False


logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = Path(__file__).parent / "config.yaml"
DEFAULT_URL = (
    "https://developer.nrel.gov/api/wind-toolkit/v2/wind/wtk-bchrrr-v1-0-0-download.csv"
)


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    """Load YAML config and merge secrets from `.env` when present."""
    path = config_path or DEFAULT_CONFIG_PATH
    if path.exists():
        load_dotenv(path.parent / ".env")
    else:
        load_dotenv()
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _nrel_settings(config: dict[str, Any]) -> dict[str, Any]:
    return config.get("nrel", {})


def get_nrel_api_key(config: dict[str, Any]) -> str:
    nrel = _nrel_settings(config)
    env_name = nrel.get("api_key_env", "NREL_API_KEY")
    key = os.environ.get(env_name, "")
    if not key:
        logger.warning("%s not set; synthetic wind data will be used", env_name)
    return key or "DEMO_KEY"


def get_nrel_email(config: dict[str, Any]) -> str:
    nrel = _nrel_settings(config)
    env_name = nrel.get("email_env", "NREL_EMAIL")
    return os.environ.get(env_name, "")


def csv_column_names(config: dict[str, Any]) -> list[str]:
    """Timestamp columns plus attribute names from config."""
    nrel = _nrel_settings(config)
    attrs = nrel.get("attributes", "windspeed_100m,winddirection_100m,temperature_100m")
    return ["Year", "Month", "Day", "Hour", "Minute"] + [a.strip() for a in attrs.split(",")]


def build_nrel_params(config: dict[str, Any], year: int) -> dict[str, str]:
    """Build NREL Wind Toolkit query params for one year (secrets from env)."""
    nrel = _nrel_settings(config)
    lat = nrel.get("lat", 41.5)
    lon = nrel.get("lon", -93.5)
    return {
        "api_key": get_nrel_api_key(config),
        "wkt": f"POINT({lon} {lat})",
        "attributes": nrel.get(
            "attributes", "windspeed_100m,winddirection_100m,temperature_100m"
        ),
        "names": str(year),
        "utc": "true" if nrel.get("utc", True) else "false",
        "leap_day": "true" if nrel.get("leap_day", False) else "false",
        "interval": str(nrel.get("interval", "60")),
        "email": get_nrel_email(config),
    }


def fetch_nrel_wind_data(
    config: dict[str, Any] | None = None,
    config_path: Path | None = None,
) -> pd.DataFrame | None:
    """Fetch hourly WTK data for years listed under ``nrel.years`` in config."""
    if config is None:
        config = load_config(config_path)
    nrel = _nrel_settings(config)
    years = nrel.get("years", [2017])
    url = nrel.get("url", DEFAULT_URL)
    timeout = int(nrel.get("timeout_seconds", 120))
    all_data: list[pd.DataFrame] = []
    for year in years:
        logger.info("   Fetching year %s...", year)
        params = build_nrel_params(config, year)
        try:
            response = requests.get(url, params=params, timeout=timeout)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.warning("Year %s fetch failed: %s", year, exc)
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
            names=csv_column_names(config),
        )
        df_year["time"] = pd.to_datetime(df_year[["Year", "Month", "Day", "Hour", "Minute"]])
        all_data.append(df_year)
        logger.info("     Fetched %s records", f"{len(df_year):,}")
    if not all_data:
        return _synthetic_wind_data()
    return pd.concat(all_data, ignore_index=True).sort_values("time")


def _synthetic_wind_data(n_hours: int = 500) -> pd.DataFrame:
    """Hourly SCADA-like wind series when NREL API is unavailable."""
    rng = np.random.default_rng(42)
    times = pd.date_range("2017-01-01", periods=n_hours, freq="h")
    speed = 8 + 3 * np.sin(np.linspace(0, 12 * np.pi, n_hours)) + rng.normal(0, 0.5, n_hours)
    return pd.DataFrame(
        {
            "Year": times.year,
            "Month": times.month,
            "Day": times.day,
            "Hour": times.hour,
            "Minute": times.minute,
            "windspeed_100m": np.clip(speed, 0, None),
            "winddirection_100m": rng.uniform(0, 360, n_hours),
            "temperature_100m": 10 + rng.normal(0, 2, n_hours),
            "time": times,
        }
    )
