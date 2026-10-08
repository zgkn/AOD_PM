"""Bias-correct the linear PM2.5 forecast against the latest station readings.

Observations come from data.gov.sg's real-time PM2.5 endpoint (same endpoint,
no API key, and retry pattern as the AQ_extrapolation repo's fetch_data.py --
reimplemented here, not imported, since the two repos are independent):

  https://api-open.data.gov.sg/v2/real-time/api/pm25
    -> items[].timestamp (ISO-8601, +08:00 / SGT)
       items[].readings.pm25_one_hourly = {north, south, east, west, central}

TIME ZONES. The CAMS forecast's valid times are UTC; the API's timestamps are
SGT (+08:00), and its ``?date=`` parameter is an SGT calendar date. All
matching here is done in *naive UTC* datetimes (API timestamps
are parsed as timezone-aware, converted to UTC, then made naive to line up
with the forecast's naive-UTC ``valid_time``). This is the same convention
build_training_table.py used when it joined the NEA SGT readings to the UTC
reanalysis. Only the dashboard's final chart spec shifts to SGT for display.

WHAT IS CORRECTED. The training targets were pm25_mean = mean of the 5 station
1-hr readings and pm25_max = max of the 5, so the matching observations are
computed the same way: mean-of-stations against the pm25_mean linear model,
max-of-stations against the pm25_max linear model. Only the two *linear*
forecasts are corrected; the ordinal band probabilities are left alone.

HOW. At every forecast valid time that already has a reading (the CAMS cycle's
reference time is always in the past, so the first few 3-hourly steps
overlap with observations), residual = observed - raw linear prediction.
The additive offset is the mean residual over all matched steps (whatever is
available -- at least one). That same constant is added to every forecast
step:

    corrected(t) = raw(t) + offset

Results are clipped at 0, and corrected max is kept >= corrected mean
(max-of-stations can never be below mean-of-stations, but the two are
corrected independently).
"""
from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("bias_correction")

API_URL = "https://api-open.data.gov.sg/v2/real-time/api/pm25"
STATIONS = ("north", "south", "east", "west", "central")

SGT = timezone(timedelta(hours=8))
SGT_OFFSET = pd.Timedelta(hours=8)

MATCH_TOLERANCE = pd.Timedelta(minutes=30)

REQUEST_TIMEOUT = 30
MAX_RETRIES = 4
DATE_REQUEST_DELAY_S = 1.5  # pace consecutive ?date= calls (data.gov.sg rate-limits bursts)


# ---------------------------------------------------------------- fetching

def _get_with_retry(params: dict) -> requests.Response:
    """GET with backoff on 429, 5xx and connection-level errors; honors Retry-After."""
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(API_URL, params=params, timeout=REQUEST_TIMEOUT)
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait = min(2 ** attempt, 30)
            log.warning("%s (attempt %d/%d) -- retrying in %ss", e.__class__.__name__, attempt, MAX_RETRIES, wait)
            time.sleep(wait)
            continue
        if resp.status_code == 429 or resp.status_code >= 500:
            if attempt == MAX_RETRIES:
                resp.raise_for_status()
            retry_after = resp.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(2 ** attempt, 30)
            log.warning("HTTP %d (attempt %d/%d) -- retrying in %ss", resp.status_code, attempt, MAX_RETRIES, wait)
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    raise RuntimeError("unreachable")


def fetch_items(sgt_date: str | None = None) -> list[dict]:
    """All items from the endpoint (latest reading if no date, else the whole
    SGT calendar day ``YYYY-MM-DD``), following ``paginationToken``."""
    params: dict = {"date": sgt_date} if sgt_date else {}
    items: list[dict] = []
    while True:
        data = _get_with_retry(params).json().get("data", {})
        items.extend(data.get("items", []))
        token = data.get("paginationToken")
        if not token:
            break
        params["paginationToken"] = token
    if sgt_date and items and not str(items[0].get("timestamp", "")).startswith(sgt_date):
        log.warning("requested date=%s but response starts at %r -- date param may have been dropped",
                    sgt_date, items[0].get("timestamp"))
    return items


def sgt_dates_between(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> list[str]:
    """SGT calendar dates (the API's ``date`` unit) touched by [start, end],
    both naive UTC. E.g. 16:00Z is already 00:00 SGT the *next* day."""
    first = (start_utc + SGT_OFFSET).date()
    last = (end_utc + SGT_OFFSET).date()
    out, d = [], first
    while d <= last:
        out.append(d.isoformat())
        d += timedelta(days=1)
    return out


def items_to_observations(items: list[dict]) -> pd.DataFrame:
    """Reduce API items to one row per reading time with station mean/max.

    Columns: obs_time (naive UTC), obs_mean, obs_max, n_stations.
    Stations with a missing value are skipped; times with none are dropped.
    """
    rows = []
    for item in items:
        ts = item.get("timestamp")
        readings = (item.get("readings") or {}).get("pm25_one_hourly") or {}
        vals = [float(readings[s]) for s in STATIONS if readings.get(s) is not None]
        if not ts or not vals:
            continue
        t = pd.to_datetime(ts, utc=True).tz_convert(None)  # SGT -> naive UTC
        rows.append({"obs_time": t, "obs_mean": float(np.mean(vals)), "obs_max": float(np.max(vals)),
                     "n_stations": len(vals)})
    if not rows:
        return pd.DataFrame(columns=["obs_time", "obs_mean", "obs_max", "n_stations"])
    df = pd.DataFrame(rows).drop_duplicates("obs_time", keep="last")
    df["obs_time"] = df["obs_time"].astype("datetime64[ns]")  # same resolution as the forecast's valid_time
    return df.sort_values("obs_time").reset_index(drop=True)


def fetch_observations(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> pd.DataFrame:
    """Every station reading from ``start_utc`` (naive UTC) onward: the
    latest reading plus each SGT calendar day covering [start, end]."""
    items: list[dict] = []
    errors: list[Exception] = []
    try:
        items += fetch_items()
    except requests.exceptions.RequestException as e:
        errors.append(e)
    for i, day in enumerate(sgt_dates_between(start_utc, end_utc)):
        if i > 0 or items:
            time.sleep(DATE_REQUEST_DELAY_S)
        try:
            items += fetch_items(day)
        except requests.exceptions.RequestException as e:
            errors.append(e)
            log.warning("giving up on date=%s: %s", day, e)
    if not items and errors:
        raise errors[-1]
    obs = items_to_observations(items)
    return obs[obs["obs_time"] >= start_utc - MATCH_TOLERANCE].reset_index(drop=True)


# -------------------------------------------------------------- correction

def apply_bias_correction(df: pd.DataFrame, obs: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Add ``pm25_linear_corrected`` / ``pm25_max_linear_corrected`` to a
    copy of ``df`` (needs naive-UTC ``valid_time``, ``pm25_linear``,
    ``pm25_max_linear``). Never raises on missing data: when no reading
    matches a valid time, the corrected columns equal the raw ones and
    ``info["status"]`` is "unavailable" with a reason."""
    out = df.copy()
    out["pm25_linear_corrected"] = out["pm25_linear"]
    out["pm25_max_linear_corrected"] = out["pm25_max_linear"]
    info: dict = {"status": "unavailable", "reason": "", "n_points": 0,
                  "offset_mean": None, "offset_max": None, "latest_reading": None,
                  "obs": pd.DataFrame()}

    if obs is None or obs.empty:
        info["reason"] = "no station readings available"
        return out, info
    info["obs"] = obs
    info["latest_reading"] = obs["obs_time"].max()

    fc = out[["valid_time", "pm25_linear", "pm25_max_linear"]].sort_values("valid_time")
    obs = obs.assign(obs_time=obs["obs_time"].astype("datetime64[ns]"))  # merge_asof needs identical dtypes
    fc = fc.assign(valid_time=fc["valid_time"].astype("datetime64[ns]"))
    matched = pd.merge_asof(
        fc, obs.sort_values("obs_time"), left_on="valid_time", right_on="obs_time",
        direction="nearest", tolerance=MATCH_TOLERANCE,
    ).dropna(subset=["obs_time"])
    if matched.empty:
        info["reason"] = "no forecast valid time has a station reading yet"
        return out, info

    res_mean = (matched["obs_mean"] - matched["pm25_linear"]).dropna()
    res_max = (matched["obs_max"] - matched["pm25_max_linear"]).dropna()
    offset_mean = float(res_mean.mean()) if len(res_mean) else 0.0
    offset_max = float(res_max.mean()) if len(res_max) else 0.0

    mean_c = (out["pm25_linear"] + offset_mean).clip(lower=0)
    max_c = (out["pm25_max_linear"] + offset_max).clip(lower=0)
    out["pm25_linear_corrected"] = mean_c
    out["pm25_max_linear_corrected"] = np.maximum(max_c, mean_c)

    info.update(status="applied", n_points=int(len(matched)), offset_mean=offset_mean,
                offset_max=offset_max)
    return out, info


def utc_now_naive() -> pd.Timestamp:
    return pd.Timestamp(datetime.now(timezone.utc)).tz_convert(None)
