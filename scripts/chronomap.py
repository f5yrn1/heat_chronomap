"""
chronomap.py — AgERA5-H version

Full Python chronomap engine with:
- AgERA5-H hourly 2m temperature (public, no credentials)
- Local time conversion
- GDD-based phenology (with "tuber initiation")
- Photoperiod-aware thermal envelopes
- Sunrise/sunset lines
- Stage bar + Month bar
- Hour x DOY climate-risk tiles

Dependencies:
pip install xarray netCDF4 pandas numpy pytz timezonefinder astral matplotlib
"""

import os
from dataclasses import dataclass
from typing import Dict, Tuple

import xarray as xr
import numpy as np
import pandas as pd

from timezonefinder import TimezoneFinder
import pytz

from astral import LocationInfo
from astral.sun import sun

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm


# ---------------------------------------------------------------------
# 1. AgERA5-H hourly temperature loader
# ---------------------------------------------------------------------

def load_agera5_hourly(lat: float, lon: float, year: int) -> pd.DataFrame:
    """
    Load AgERA5-H hourly 2m temperature for a single point.
    Public, no credentials required.
    """

    # Public AgERA5-H bucket (example structure)
    url = f"https://objectstore.eea.europa.eu/agera5/t2m_hourly/{year}.nc"

    ds = xr.open_dataset(url)

    # Select nearest grid cell
    ds_point = ds.sel(latitude=lat, longitude=lon, method="nearest")

    # Convert to Celsius
    temp_c = ds_point["t2m"].values - 273.15
    time = pd.to_datetime(ds_point["time"].values)

    return pd.DataFrame({"timestamp_utc": time, "temp": temp_c})


# ---------------------------------------------------------------------
# 2. Timezone + local-time conversion
# ---------------------------------------------------------------------

def get_local_timezone(lat: float, lon: float) -> str:
    tf = TimezoneFinder()
    tz_name = tf.timezone_at(lat=lat, lng=lon)
    return tz_name if tz_name else "UTC"


def to_local_time(df: pd.DataFrame, lat: float, lon: float) -> pd.DataFrame:
    tz_name = get_local_timezone(lat, lon)
    local_tz = pytz.timezone(tz_name)

    df = df.copy()
    df["timestamp"] = (
        df["timestamp_utc"]
        .dt.tz_localize("UTC")
        .dt.tz_convert(local_tz)
    )
    df["tz"] = tz_name
    return df


# ---------------------------------------------------------------------
# 3. GDD-based phenology windows
# ---------------------------------------------------------------------

@dataclass
class StageWindows:
    emergence: Tuple[int, int]
    vegetative: Tuple[int, int]
    tuber_initiation: Tuple[int, int]
    bulking: Tuple[int, int]
    maturation: Tuple[int, int]
    senescence: Tuple[int, int]


def compute_stage_windows(weather: pd.DataFrame, planting_date: pd.Timestamp, tbase: float) -> StageWindows:
    df = weather.copy()
    df = df[df["timestamp"].dt.date >= planting_date.date()].copy()
    df["doy"] = df["timestamp"].dt.dayofyear

    df["gdd"] = np.maximum(0, df["temp"] - tbase) / 24.0
    df["cum_gdd"] = df["gdd"].cumsum()

    thresholds = {
        "emergence": 150,
        "vegetative": 350,
        "tuber_initiation": 550,
        "bulking": 900,
        "maturation": 1200,
    }

    stage_doys = {}
    for stage, th in thresholds.items():
        idx = np.where(df["cum_gdd"].values >= th)[0]
        stage_doys[stage] = int(df["doy"].iloc[idx[0]]) if len(idx) else np.nan

    last_doy = int(weather["timestamp"].dt.dayofyear.max())
    for k, v in stage_doys.items():
        if np.isnan(v):
            stage_doys[k] = last_doy

    return StageWindows(
        emergence=(int(planting_date.dayofyear), stage_doys["emergence"]),
        vegetative=(stage_doys["emergence"], stage_doys["vegetative"]),
        tuber_initiation=(stage_doys["vegetative"], stage_doys["tuber_initiation"]),
        bulking=(stage_doys["tuber_initiation"], stage_doys["bulking"]),
        maturation=(stage_doys["bulking"], stage_doys["maturation"]),
        senescence=(stage_doys["maturation"], last_doy),
    )


def assign_stage(doy: int, windows: StageWindows) -> str:
    for name, (start, end) in windows.__dict__.items():
        if start <= doy <= end:
            return name.replace("_", " ")
    return None


# ---------------------------------------------------------------------
# 4. Photoperiod + envelopes + risk classification
# ---------------------------------------------------------------------

THERMAL_ENVELOPES = {
    "emergence": {
        "day":   {"low": 12, "opt_low": 14, "opt_high": 20, "high": 26},
        "night": {"low": 8,  "opt_low": 10, "opt_high": 16, "high": 20},
    },
    "vegetative": {
        "day":   {"low": 14, "opt_low": 18, "opt_high": 24, "high": 30},
        "night": {"low": 10, "opt_low": 12, "opt_high": 18, "high": 22},
    },
    "tuber initiation": {
        "day":   {"low": 12, "opt_low": 15, "opt_high": 22, "high": 28},
        "night": {"low": 8,  "opt_low": 10, "opt_high": 16, "high": 20},
    },
    "bulking": {
        "day":   {"low": 12, "opt_low": 15, "opt_high": 20, "high": 27},
        "night": {"low": 8,  "opt_low": 12, "opt_high": 16, "high": 22},
    },
    "maturation": {
        "day":   {"low": 10, "opt_low": 14, "opt_high": 18, "high": 25},
        "night": {"low": 6,  "opt_low": 10, "opt_high": 14, "high": 18},
    },
    "senescence": {
        "day":   {"low": 5,  "opt_low": 8,  "opt_high": 15, "high": 25},
        "night": {"low": 2,  "opt_low": 5,  "opt_high": 10, "high": 15},
    },
}


def compute_sun_times_for_dates(lat: float, lon: float, tz_name: str, dates: pd.DatetimeIndex) -> pd.DataFrame:
    location = LocationInfo(latitude=lat, longitude=lon, timezone=tz_name)
    out = []
    for d in dates:
        s_times = sun(location.observer, date=d.date(), tzinfo=pytz.timezone(tz_name))
        out.append({"date": d.date(), "sunrise": s_times["sunrise"], "sunset": s_times["sunset"]})
    return pd.DataFrame(out)


def classify_risk(temp: float, stage: str, photoperiod: str) -> str:
    if stage is None:
        return None
    if temp < 0:
        return "frost"

    env = THERMAL_ENVELOPES.get(stage, {}).get(photoperiod)
    if env is None:
        return None

    if temp < env["low"]:
        return "cold"
    if temp > env["high"]:
        return "heat"
    if temp < env["opt_low"]:
        return "cool"
    if temp > env["opt_high"]:
        return "warm"
    return "optimal"


# ---------------------------------------------------------------------
# 5. Chronomap generation
# ---------------------------------------------------------------------

RISK_LEVELS = ["frost", "cold", "cool", "optimal", "warm", "heat"]
RISK_COLORS = {
    "frost": "purple4",
    "cold": "steelblue",
    "cool": "lightblue",
    "optimal": "green",
    "warm": "orange",
    "heat": "red",
}


def generate_chronomap(
    weather: pd.DataFrame,
    lat: float,
    lon: float,
    planting_date: str,
    harvest_date: str,
    tbase: float = 7.0,
    title: str = "Photoperiod-Aware Hourly Climate-Risk Chronomap",
) -> plt.Figure:

    df = weather.copy()

    # Filter by planting/harvest
    planting_date = pd.to_datetime(planting_date).tz_localize(df["timestamp"].dt.tz)
    harvest_date = pd.to_datetime(harvest_date).tz_localize(df["timestamp"].dt.tz)
    df = df[(df["timestamp"] >= planting_date) & (df["timestamp"] <= harvest_date)].copy()

    df["date"] = df["timestamp"].dt.date
    df["doy"] = df["timestamp"].dt.dayofyear
    df["hour"] = df["timestamp"].dt.hour

    # Sunrise/sunset
    tz_name = df["tz"].iloc[0]
    unique_dates = pd.to_datetime(sorted(df["date"].unique()))
    sun_df = compute_sun_times_for_dates(lat, lon, tz_name, unique_dates)
    df = df.merge(sun_df, on="date", how="left")

    df["is_day"] = (df["timestamp"] >= df["sunrise"]) & (df["timestamp"] < df["sunset"])
    df["photoperiod"] = np.where(df["is_day"], "day", "night")

    # Stage windows
    stage_windows = compute_stage_windows(df, planting_date, tbase)
    df["stage"] = df["doy"].apply(lambda d: assign_stage(d, stage_windows))

    # Risk classification
    df["risk"] = [
        classify_risk(t, s, p)
        for t, s, p in zip(df["temp"], df["stage"], df["photoperiod"])
    ]

    # Stage bar
    stage_bar = (
        df.groupby("stage")
          .agg(doy_min=("doy", "min"), doy_max=("doy", "max"))
          .reset_index()
    )

    stage_order = [
        "emergence",
        "vegetative",
        "tuber initiation",
        "bulking",
        "maturation",
        "senescence"
    ]

    stage_colors = {
        "emergence": "#b2df8a",
        "vegetative": "#33a02c",
        "tuber initiation": "#1f78b4",
        "bulking": "#a6cee3",
        "maturation": "#fb9a99",
        "senescence": "#e31a1c"
    }

    stage_bar["stage"] = pd.Categorical(stage_bar["stage"], categories=stage_order, ordered=True)
    stage_bar = stage_bar.sort_values("stage")

    # Month bar
    month_bar = (
        df.groupby(df["timestamp"].dt.month)
          .agg(doy_min=("doy", "min"), doy_max=("doy", "max"))
          .reset_index()
    )
    month_bar["month_label"] = month_bar["timestamp"].dt.month_name().str[:3]

    # Build risk grid
    min_doy = df["doy"].min()
    max_doy = df["doy"].max()
    doys = np.arange(min_doy, max_doy + 1)
    hours = np.arange(0, 24)

    risk_to_int = {r: i for i, r in enumerate(RISK_LEVELS)}
    grid = np.full((len(doys), len(hours)), np.nan)

    for _, row in df.iterrows():
        i = row["doy"] - min_doy
        j = row["hour"]
        r = row["risk"]
        if r in risk_to_int:
            grid[i, j] = risk_to_int[r]

    # Sunrise/sunset per DOY
    sun_daily = (
        df.groupby("doy")
        .agg(
            sunrise_hour=("sunrise", lambda x: x.iloc[0].hour if pd.notnull(x.iloc[0]) else np.nan),
            sunset_hour=("sunset", lambda x: x.iloc[0].hour if pd.notnull(x.iloc[0]) else np.nan),
        )
        .reset_index()
    )

    # -----------------------------------------------------------------
    # Plot layout
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(12, 8))

    # Main chronomap axes
    ax = fig.add_axes([0.25, 0.1, 0.70, 0.8])

    cmap = ListedColormap([RISK_COLORS[r] for r in RISK_LEVELS])
    bounds = np.arange(len(RISK_LEVELS) + 1) - 0.5
    norm = BoundaryNorm(bounds, cmap.N)

    im = ax.imshow(
        grid,
        aspect="auto",
        origin="upper",
        cmap=cmap,
        norm=norm,
        extent=[-0.5, 23.5, max_doy + 0.5, min_doy - 0.5],
    )

    # Sunrise/sunset lines
    for _, row in sun_daily.iterrows():
        doy = row["doy"]
        if not np.isnan(row["sunrise_hour"]):
            ax.plot([row["sunrise_hour"], row["sunrise_hour"]], [doy - 0.5, doy + 0.5], color="black", linewidth=1.2)
        if not np.isnan(row["sunset_hour"]):
            ax.plot([row["sunset_hour"], row["sunset_hour"]], [doy - 0.5, doy + 0.5], color="black", linewidth=1.2)

    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Day of Year")
    ax.set_title(f"{title}\nTimezone: {tz_name}")

    # Legend
    cbar = fig.colorbar(im, ax=ax, ticks=np.arange(len(RISK_LEVELS)))
    cbar.ax.set_yticklabels(RISK_LEVELS)
    cbar.set_label("Thermal Envelope")

    # -----------------------------------------------------------------
    # Stage bar
    # -----------------------------------------------------------------
    stage_ax = fig.add_axes([0.05, 0.1, 0.07, 0.8])

    for _, row in stage_bar.iterrows():
        stage_ax.fill_between(
            x=[0, 1],
            y1=row["doy_min"],
            y2=row["doy_max"],
            color=stage_colors[row["stage"]],
            edgecolor="black",
            linewidth=0.5
        )
        stage_ax.text(
            0.5,
            (row["doy_min"] + row["doy_max"]) / 2,
            row["stage"],
            ha="center",
            va="center",
            rotation=90,
            fontsize=8
        )

    stage_ax.set_ylim(ax.get_ylim())
    stage_ax.set_xticks([])
    stage_ax.set_yticks([])
    stage_ax.set_title("Stage", fontsize=9)

    # -----------------------------------------------------------------
    # Month bar
    # -----------------------------------------------------------------
    month_ax = fig.add_axes([0.14, 0.1, 0.07, 0.8])

    for _, row in month_bar.iterrows():
        month_ax.fill_between(
            x=[0, 1],
            y1=row["doy_min"],
            y2=row["doy_max"],
            color="lightgray",
            edgecolor="black",
            linewidth=0.5
        )
        month_ax.text(
            0.5,
            (row["doy_min"] + row["doy_max"]) / 2,
            row["month_label"],
            ha="center",
            va="center",
            fontsize=8
        )

    month_ax.set_ylim(ax.get_ylim())
    month_ax.set_xticks([])
    month_ax.set_yticks([])
    month_ax.set_title("Month", fontsize=9)

    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------
# 6. End-to-end pipeline
# ---------------------------------------------------------------------

def build_chronomap_from_agera5(
    lat: float,
    lon: float,
    year: int,
    planting_date: str,
    harvest_date: str,
    tbase: float = 7.0,
) -> plt.Figure:

    df_utc = load_agera5_hourly(lat, lon, year)
    df_local = to_local_time(df_utc, lat, lon)

    fig = generate_chronomap(
        weather=df_local,
        lat=lat,
        lon=lon,
        planting_date=planting_date,
        harvest_date=harvest_date,
        tbase=tbase,
    )
    return fig


if __name__ == "__main__":
    # Example manual run
    fig = build_chronomap_from_agera5(
        lat=50.067,
        lon=-112.097,
        year=2024,
        planting_date="2024-05-20",
        harvest_date="2024-09-30",
    )
    fig.savefig("chronomap.png", dpi=150)
