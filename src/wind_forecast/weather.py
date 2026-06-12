"""Open-Meteo historical weather forecast ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import urlopen

import pandas as pd


BASE_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"


@dataclass(frozen=True)
class WeatherLocation:
    name: str
    latitude: float
    longitude: float
    note: str


WEATHER_LOCATIONS = [
    WeatherLocation("north_sea", 54.5, 7.5, "North Sea offshore wind proxy"),
    WeatherLocation("schleswig_holstein", 54.2, 9.6, "Northern onshore wind proxy"),
    WeatherLocation("lower_saxony", 53.1, 8.2, "North-western onshore wind proxy"),
    WeatherLocation("brandenburg", 52.4, 13.1, "Eastern onshore wind proxy"),
]

HOURLY_VARIABLES = [
    "wind_speed_100m",
    "wind_direction_100m",
    "wind_gusts_10m",
    "temperature_2m",
    "pressure_msl",
]


def _read_json(url: str) -> dict | list[dict]:
    with urlopen(url, timeout=90) as response:
        return json.loads(response.read().decode("utf-8"))


def _weather_url(start: str, end: str) -> str:
    params = {
        "latitude": ",".join(str(location.latitude) for location in WEATHER_LOCATIONS),
        "longitude": ",".join(str(location.longitude) for location in WEATHER_LOCATIONS),
        "start_date": start,
        "end_date": end,
        "hourly": ",".join(HOURLY_VARIABLES),
        "timezone": "UTC",
        "wind_speed_unit": "ms",
        "models": "best_match",
        "cell_selection": "nearest",
    }
    return f"{BASE_URL}?{urlencode(params)}"


def _frame_for_location(payload: dict, location: WeatherLocation) -> pd.DataFrame:
    hourly = payload["hourly"]
    df = pd.DataFrame({"timestamp_utc": pd.to_datetime(hourly["time"], utc=True)})
    prefix = location.name
    for variable in HOURLY_VARIABLES:
        raw = pd.Series(hourly[variable], dtype="float64")
        suffix = {
            "wind_speed_100m": "wind_speed_100m_ms",
            "wind_direction_100m": "wind_direction_100m_deg",
            "wind_gusts_10m": "wind_gusts_10m_ms",
            "temperature_2m": "temperature_2m_c",
            "pressure_msl": "pressure_msl_hpa",
        }[variable]
        df[f"{prefix}_{suffix}"] = raw
    return df


def fetch_weather_forecasts(start: str, end: str) -> pd.DataFrame:
    """Fetch archived weather forecast features for representative wind locations."""

    payload = _read_json(_weather_url(start, end))
    if isinstance(payload, dict):
        payloads = [payload]
    else:
        payloads = payload
    if len(payloads) != len(WEATHER_LOCATIONS):
        raise RuntimeError(f"Expected {len(WEATHER_LOCATIONS)} weather payloads, got {len(payloads)}")

    merged: pd.DataFrame | None = None
    for location, location_payload in zip(WEATHER_LOCATIONS, payloads):
        df = _frame_for_location(location_payload, location)
        merged = df if merged is None else merged.merge(df, on="timestamp_utc", how="outer")

    assert merged is not None
    return merged.sort_values("timestamp_utc").reset_index(drop=True)


def build_weather_dataset(
    start: str,
    end: str,
    raw_dir: Path,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    raw_dir.mkdir(parents=True, exist_ok=True)
    raw_path = raw_dir / "open_meteo_historical_forecast.csv"
    if use_cache and raw_path.exists():
        df = pd.read_csv(raw_path)
        df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
        start_utc = pd.Timestamp(start, tz="UTC")
        end_utc = pd.Timestamp(end, tz="UTC") + pd.Timedelta(hours=23)
        if df["timestamp_utc"].min() <= start_utc and df["timestamp_utc"].max() >= end_utc:
            return df[(df["timestamp_utc"] >= start_utc) & (df["timestamp_utc"] <= end_utc)].reset_index(drop=True)

    df = fetch_weather_forecasts(start, end)
    df.to_csv(raw_path, index=False)
    return df
