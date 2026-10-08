"""Offline tests for scripts/bias_correction.py (no network).

Run: python tests/test_bias_correction.py   (or pytest)
"""
import math
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import bias_correction as bc  # noqa: E402


def item(ts_sgt, **stations):
    return {"timestamp": ts_sgt, "readings": {"pm25_one_hourly": stations}}


# Shape of a data.gov.sg /v2/real-time/api/pm25 response's data.items
ITEMS = [
    item("2026-10-08T08:00:00+08:00", north=10, south=20, east=30, west=40, central=50),   # = 00:00Z
    item("2026-10-08T09:00:00+08:00", north=1, south=1, east=1, west=1, central=1),        # not a 3-hourly step
    item("2026-10-08T11:00:00+08:00", north=20, south=30, east=40, west=50, central=None),  # = 03:00Z, 4 stations
]


def forecast(hours=(0, 3, 6, 9, 12, 48)):
    t0 = pd.Timestamp("2026-10-08 00:00")  # naive UTC
    return pd.DataFrame({
        # ns resolution (what xarray yields) vs the API-parsed observations' resolution
        "valid_time": pd.Series([t0 + pd.Timedelta(hours=h) for h in hours]).astype("datetime64[ns]"),
        "pm25_linear": [20.0] * len(hours),
        "pm25_max_linear": [30.0] * len(hours),
    })


def test_sgt_timestamps_become_naive_utc():
    obs = bc.items_to_observations(ITEMS)
    assert list(obs["obs_time"][:2]) == [pd.Timestamp("2026-10-08 00:00"), pd.Timestamp("2026-10-08 01:00")]
    assert obs["obs_time"].dt.tz is None
    first = obs.iloc[0]
    assert first["obs_mean"] == 30 and first["obs_max"] == 50
    last = obs.iloc[2]  # missing station skipped, not treated as 0
    assert last["obs_mean"] == 35 and last["obs_max"] == 50 and last["n_stations"] == 4


def test_sgt_dates_cross_midnight():
    # 16:00Z on the 8th is 00:00 SGT on the 9th
    days = bc.sgt_dates_between(pd.Timestamp("2026-10-08 10:00"), pd.Timestamp("2026-10-08 17:00"))
    assert days == ["2026-10-08", "2026-10-09"]


def test_offset_and_decay():
    obs = bc.items_to_observations(ITEMS)
    out, info = bc.apply_bias_correction(forecast(), obs, tau_hours=12)
    # matched steps 00Z (resid mean +10, max +20) and 03Z (+15, +20) -> offsets 12.5 / 20
    assert info["status"] == "applied" and info["n_points"] == 2
    assert math.isclose(info["offset_mean"], 12.5) and math.isclose(info["offset_max"], 20.0)
    assert info["anchor"] == pd.Timestamp("2026-10-08 03:00")
    c = out.set_index("valid_time")["pm25_linear_corrected"]
    assert math.isclose(c[pd.Timestamp("2026-10-08 00:00")], 32.5)            # before anchor: full offset
    assert math.isclose(c[pd.Timestamp("2026-10-08 03:00")], 32.5)            # at anchor: full offset
    assert math.isclose(c[pd.Timestamp("2026-10-08 06:00")], 20 + 12.5 * math.exp(-3 / 12))
    assert math.isclose(c[pd.Timestamp("2026-10-08 12:00")], 20 + 12.5 * math.exp(-9 / 12))
    assert math.isclose(c[pd.Timestamp("2026-10-10 00:00")], 20 + 12.5 * math.exp(-45 / 12))  # day 2: ~raw
    assert (out["pm25_max_linear_corrected"] >= out["pm25_linear_corrected"]).all()


def test_clip_and_max_not_below_mean():
    obs = bc.items_to_observations([item("2026-10-08T08:00:00+08:00", north=0, south=0, east=0, west=0, central=0)])
    df = forecast((0, 3))
    out, info = bc.apply_bias_correction(df, obs)
    assert (out["pm25_linear_corrected"] >= 0).all()           # 20 - 20 = 0, never negative
    # mean offset +big, max offset small -> max lifted to mean
    obs2 = bc.items_to_observations([item("2026-10-08T08:00:00+08:00", north=60, south=0, east=0, west=0, central=0)])
    df2 = forecast((0,)); df2["pm25_max_linear"] = 21.0
    out2, _ = bc.apply_bias_correction(df2, obs2)
    assert out2["pm25_max_linear_corrected"].iloc[0] >= out2["pm25_linear_corrected"].iloc[0]


def test_graceful_without_readings():
    for obs in (None, pd.DataFrame(columns=["obs_time", "obs_mean", "obs_max", "n_stations"])):
        out, info = bc.apply_bias_correction(forecast(), obs)
        assert info["status"] == "unavailable" and info["reason"]
        assert (out["pm25_linear_corrected"] == out["pm25_linear"]).all()
    # readings exist but none within tolerance of any valid time
    late = bc.items_to_observations([item("2026-10-08T09:40:00+08:00", north=5, south=5, east=5, west=5, central=5)])
    _, info = bc.apply_bias_correction(forecast(), late)
    assert info["status"] == "unavailable"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
