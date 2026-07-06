"""SMARD public API ingestion for German wind generation."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen
from zoneinfo import ZoneInfo

import pandas as pd


BASE_URL = "https://www.smard.de/app/chart_data"
MARKET_TZ = ZoneInfo("Europe/Berlin")


@dataclass(frozen=True)
class SmardSeries:
    name: str
    filter_id: int
    region: str
    unit: str
    description: str


SERIES = [
    SmardSeries(
        name="actual_wind_offshore_mw",
        filter_id=1225,
        region="DE",
        unit="MW",
        description="Actual offshore wind generation",
    ),
    SmardSeries(
        name="actual_wind_onshore_mw",
        filter_id=4067,
        region="DE",
        unit="MW",
        description="Actual onshore wind generation",
    ),
    SmardSeries(
        name="forecast_wind_onshore_mw",
        filter_id=123,
        region="DE",
        unit="MW",
        description="Forecast onshore wind generation",
    ),
    SmardSeries(
        name="forecast_wind_offshore_mw",
        filter_id=3791,
        region="DE",
        unit="MW",
        description="Forecast offshore wind generation",
    ),
    SmardSeries(
        name="price_da_eur_mwh",
        filter_id=4169,
        region="DE-LU",
        unit="EUR/MWh",
        description="Day-ahead auction price",
    ),
]


def _read_json(url: str, *, attempts: int = 3, timeout: int = 90) -> dict:
    last_error: Exception | None = None
    for attempt in range(attempts):
        try:
            with urlopen(url, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except Exception as exc:
            last_error = exc
            if attempt < attempts - 1:
                time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"Failed to read SMARD URL after {attempts} attempts: {url}") from last_error


def _to_date(value: str | date | datetime) -> date:
    if isinstance(value, str):
        return datetime.fromisoformat(value).date()
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return value.astimezone(MARKET_TZ).date() if value.tzinfo else value.date()


def _local_midnight_ms(value: str | date | datetime) -> int:
    local_date = _to_date(value)
    dt = datetime(local_date.year, local_date.month, local_date.day, tzinfo=MARKET_TZ)
    return int(dt.timestamp() * 1000)


def _local_end_of_day_ms(value: str | date | datetime) -> int:
    local_date = _to_date(value)
    next_midnight = datetime(local_date.year, local_date.month, local_date.day, tzinfo=MARKET_TZ) + timedelta(days=1)
    return int(next_midnight.timestamp() * 1000) - 1


def _timestamp_index(series: SmardSeries, resolution: str = "hour") -> list[int]:
    url = f"{BASE_URL}/{series.filter_id}/{series.region}/index_{resolution}.json"
    return _read_json(url)["timestamps"]


def _chunk(series: SmardSeries, timestamp: int, resolution: str = "hour") -> list[list[float | int | None]]:
    url = (
        f"{BASE_URL}/{series.filter_id}/{series.region}/"
        f"{series.filter_id}_{series.region}_{resolution}_{timestamp}.json"
    )
    return _read_json(url).get("series", [])


def _candidate_timestamps(timestamps: Iterable[int], start_ms: int, end_ms: int) -> list[int]:
    # SMARD chunks are weekly-ish. Include the chunk before the requested
    # start so edge hours are not lost when the chunk boundary is earlier.
    ordered = sorted(timestamps)
    lookback_ms = 8 * 24 * 3600 * 1000
    return [ts for ts in ordered if start_ms - lookback_ms <= ts <= end_ms]


def fetch_smard_series(
    series: SmardSeries,
    start: str,
    end: str,
    *,
    sleep_seconds: float = 0.02,
) -> pd.DataFrame:
    """Fetch one hourly SMARD series for a local German date range."""

    start_ms = _local_midnight_ms(start)
    end_ms = _local_end_of_day_ms(end)
    timestamps = _candidate_timestamps(_timestamp_index(series), start_ms, end_ms)

    rows: list[list[float | int | None]] = []
    for timestamp in timestamps:
        rows.extend(_chunk(series, timestamp))
        if sleep_seconds:
            time.sleep(sleep_seconds)

    df = pd.DataFrame(rows, columns=["timestamp_ms", series.name])
    if df.empty:
        raise RuntimeError(f"No SMARD rows returned for {series.name} between {start} and {end}")

    df = df.drop_duplicates("timestamp_ms").sort_values("timestamp_ms")
    df = df[(df["timestamp_ms"] >= start_ms) & (df["timestamp_ms"] <= end_ms)]
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_ms"], unit="ms", utc=True)
    return df[["timestamp_utc", series.name]]


def _read_cached_series(path: Path, column: str, start: str, end: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"], utc=True)
    start_utc = pd.to_datetime(_local_midnight_ms(start), unit="ms", utc=True)
    end_utc = pd.to_datetime(_local_end_of_day_ms(end), unit="ms", utc=True)
    if df["timestamp_utc"].min() > start_utc or df["timestamp_utc"].max() < end_utc.floor("h"):
        raise ValueError("Cached SMARD series does not cover requested date range.")
    df = df[(df["timestamp_utc"] >= start_utc) & (df["timestamp_utc"] <= end_utc)]
    return df[["timestamp_utc", column]]


def _regularize_hourly_index(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("timestamp_utc").set_index("timestamp_utc")
    full_index = pd.date_range(df.index.min(), df.index.max(), freq="h", tz=timezone.utc)
    df = df.reindex(full_index)
    df.index.name = "timestamp_utc"

    forecast_columns = [column for column in df.columns if column.startswith("forecast_")]
    df[forecast_columns] = df[forecast_columns].interpolate(method="time", limit=2, limit_direction="both")
    df = df.reset_index()
    df["timestamp_local"] = df["timestamp_utc"].dt.tz_convert(MARKET_TZ)
    df["actual_wind_total_mw"] = df["actual_wind_onshore_mw"] + df["actual_wind_offshore_mw"]
    df["forecast_wind_total_mw"] = df["forecast_wind_onshore_mw"] + df["forecast_wind_offshore_mw"]
    return df


def build_smard_dataset(
    start: str,
    end: str,
    raw_dir: Path,
    *,
    use_cache: bool = True,
) -> pd.DataFrame:
    """Fetch and merge all SMARD wind series."""

    raw_dir.mkdir(parents=True, exist_ok=True)
    merged: pd.DataFrame | None = None

    for series in SERIES:
        raw_path = raw_dir / f"{series.name}.csv"
        if use_cache and raw_path.exists():
            try:
                df = _read_cached_series(raw_path, series.name, start, end)
            except ValueError:
                df = fetch_smard_series(series, start, end)
                df.to_csv(raw_path, index=False)
        else:
            df = fetch_smard_series(series, start, end)
            df.to_csv(raw_path, index=False)
        merged = df if merged is None else merged.merge(df, on="timestamp_utc", how="outer")

    assert merged is not None
    return _regularize_hourly_index(merged)
