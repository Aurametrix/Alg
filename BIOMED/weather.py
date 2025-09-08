# weather analysis 
#
#  this cell:
# - Defines robust functions to read daily fixed-width data and monthly normals
# - Aggregates to monthly and weekly periods
# - Computes rolling N-day summaries
# - Detects extreme events (unit-aware, °F & inches) with proper MONTH merge
# - Computes linear trends using scipy.stats.linregress when available (fallback to slope-only if not)
# - Runs an example end-to-end if your input files exist in /datadirectory/data
#
# Inputs expected (place these in /datadirectory/data or change the paths below):
#   - /datadirectory/data/4109539.txt
#   - /datadirectory/data/normals-monthly-2006-2020-2025-08-31T23-21-51.csv
#
# Outputs (written to /datadirectory/data/outputs):
#   - monthly_deviations.csv
#   - weekly_deviations.csv
#   - triple_day_summaries.csv
#   - critical_events.csv
#   - monthly_trends.csv
#
import os
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd

# ----------------------------
# Readers
# ----------------------------

def read_daily_fixed_width(path: str) -> pd.DataFrame:
    """Read daily weather data from a fixed-width file (NCEI GHCND daily style)."""
    # header row, dashed row, then data
    try:
        # skip dashed line at row index 1
        df = pd.read_fwf(path, header=0, skiprows=[1], infer_nrows=500)
    except Exception as e:
        raise ValueError(f"Failed to read fixed-width file: {e}")

    # Normalize station code and parse date
    df["STATION"] = (
        df["STATION"].astype(str)
        .str.replace(r"^GHCND:", "", regex=True)
        .str.strip()
    )
    df["DATE"] = pd.to_datetime(df["DATE"].astype(str), format="%Y%m%d", errors="coerce")

    # Convert numeric columns and handle -9999 as NaN
    for col in df.columns:
        if col not in ["STATION", "STATION_NAME", "DATE"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    num_cols = df.columns.difference(["STATION", "STATION_NAME", "DATE"])
    df[num_cols] = df[num_cols].replace(-9999, np.nan)

    # Drop invalid dates
    df = df.dropna(subset=["DATE"])

    return df


def read_normals_csv(path: str) -> pd.DataFrame:
    """Read monthly normals CSV; returns columns: STATION, MONTH, *_NORMAL."""
    try:
        normals = pd.read_csv(path)
    except Exception as e:
        raise ValueError(f"Failed to read normals CSV: {e}")

    rename_map = {
        "MLY-PRCP-NORMAL": "PRCP_NORMAL",
        "MLY-SNOW-NORMAL": "SNOW_NORMAL",
        "MLY-TAVG-NORMAL": "TAVG_NORMAL",
        "MLY-TMAX-NORMAL": "TMAX_NORMAL",
        "MLY-TMIN-NORMAL": "TMIN_NORMAL",
        "DATE": "MONTH",
    }
    normals = normals.rename(columns=rename_map)
    normals["MONTH"] = pd.to_numeric(normals["MONTH"], errors="coerce").astype("Int64")
    normals["STATION"] = (
        normals["STATION"].astype(str)
        .str.replace(r"^GHCND:", "", regex=True)
        .str.strip()
    )

    keep = ["STATION", "MONTH", "PRCP_NORMAL", "SNOW_NORMAL", "TAVG_NORMAL", "TMAX_NORMAL", "TMIN_NORMAL"]
    missing = [c for c in keep if c not in normals.columns]
    if missing:
        raise ValueError(f"Normals CSV missing expected columns: {missing}")

    return normals[keep]


# ----------------------------
# Aggregators
# ----------------------------

def aggregate_monthly(daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate daily observations to monthly totals/means."""
    d = daily.copy()
    d["YEAR"] = d["DATE"].dt.year
    d["MONTH"] = d["DATE"].dt.month
    monthly = (
        d.groupby(["STATION", "YEAR", "MONTH"], dropna=False)
         .agg(
            PRCP_MONTH=("PRCP", "sum"),
            SNOW_MONTH=("SNOW", "sum"),
            TAVG_MONTH=("TAVG", "mean"),
            TMAX_MONTH=("TMAX", "mean"),
            TMIN_MONTH=("TMIN", "mean"),
            N_DAYS=("DATE", "nunique"),
         )
         .reset_index()
    )
    return monthly


def aggregate_weekly(daily: pd.DataFrame, week_freq: str = "W-MON") -> pd.DataFrame:
    """Weekly aggregation (weeks start on Monday by default)."""
    d = daily.copy()
    weekly = d.groupby(["STATION", pd.Grouper(key="DATE", freq=week_freq)]) \
              .agg(
                    PRCP_WEEK=("PRCP", "sum"),
                    SNOW_WEEK=("SNOW", "sum"),
                    TAVG_WEEK=("TAVG", "mean"),
                    TMAX_WEEK=("TMAX", "mean"),
                    TMIN_WEEK=("TMIN", "mean"),
                    N_DAYS=("DATE", "nunique"),
               ).reset_index()

    weekly["YEAR"] = weekly["DATE"].dt.year
    weekly["WEEK"] = weekly["DATE"].dt.isocalendar().week.astype(int)
    weekly["MONTH"] = weekly["DATE"].dt.month  # For merge with monthly normals (approximation)

    return weekly


def aggregate_rolling(daily: pd.DataFrame, window: int = 3) -> pd.DataFrame:
    """Rolling N-day window summaries (sum for precip/snow; mean for temps)."""
    d = daily.sort_values(["STATION", "DATE"]).copy()
    g = d.groupby("STATION")
    d[f"PRCP_{window}D"] = g["PRCP"].rolling(window=window, min_periods=1).sum().reset_index(0, drop=True)
    d[f"SNOW_{window}D"] = g["SNOW"].rolling(window=window, min_periods=1).sum().reset_index(0, drop=True)
    d[f"TAVG_{window}D"] = g["TAVG"].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
    d[f"TMAX_{window}D"] = g["TMAX"].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
    d[f"TMIN_{window}D"] = g["TMIN"].rolling(window=window, min_periods=1).mean().reset_index(0, drop=True)
    d[f"N_DAYS_{window}D"] = window
    d["YEAR"] = d["DATE"].dt.year
    d["MONTH"] = d["DATE"].dt.month
    cols = ["STATION", "DATE", "YEAR", "MONTH",
            f"PRCP_{window}D", f"SNOW_{window}D", f"TAVG_{window}D", f"TMAX_{window}D", f"TMIN_{window}D", f"N_DAYS_{window}D"]
    return d[cols]


# ----------------------------
# Deviations vs normals
# ----------------------------

def compute_deviations(period_df: pd.DataFrame, normals: pd.DataFrame, period_type: str = "monthly") -> pd.DataFrame:
    """
    Compute deviations for monthly or weekly aggregates.
    - monthly: join normals on ["STATION","MONTH"]
    - weekly:  join normals on ["STATION","MONTH"] (approximate weekly normals by monthly)
    """
    if period_type not in {"monthly", "weekly"}:
        raise ValueError("period_type must be 'monthly' or 'weekly'")

    joined = period_df.merge(normals, on=["STATION", "MONTH"], how="left")

    if period_type == "monthly":
        mapping = {
            "PRCP": ("PRCP_MONTH", "PRCP_NORMAL"),
            "SNOW": ("SNOW_MONTH", "SNOW_NORMAL"),
            "TAVG": ("TAVG_MONTH", "TAVG_NORMAL"),
            "TMAX": ("TMAX_MONTH", "TMAX_NORMAL"),
            "TMIN": ("TMIN_MONTH", "TMIN_NORMAL"),
        }
    else:  # weekly
        mapping = {
            "PRCP": ("PRCP_WEEK", "PRCP_NORMAL"),
            "SNOW": ("SNOW_WEEK", "SNOW_NORMAL"),
            "TAVG": ("TAVG_WEEK", "TAVG_NORMAL"),
            "TMAX": ("TMAX_WEEK", "TMAX_NORMAL"),
            "TMIN": ("TMIN_WEEK", "TMIN_NORMAL"),
        }

    for key, (obs_col, norm_col) in mapping.items():
        joined[f"DEV_{key}"] = joined[obs_col] - joined[norm_col]

    # Column ordering
    base = ["STATION", "YEAR", "MONTH"]
    if period_type == "weekly":
        base = ["STATION", "YEAR", "WEEK", "MONTH"]

    metrics = []
    for obs_col, norm_col in mapping.values():
        metrics.extend([obs_col, norm_col, f"DEV_{obs_col.split('_')[0].upper()}"])

    # Add day count if present
    if "N_DAYS" in joined.columns:
        base.append("N_DAYS")

    return joined[base + metrics]


# ----------------------------
# Extreme event detection (unit-aware: °F & inches)
# ----------------------------

def detect_critical_events(daily: pd.DataFrame, normals: pd.DataFrame,
                           rel_temp_thresh_f: float = 5.0,
                           abs_hot_f: float = 95.0,
                           abs_cold_f: float = 20.0,
                           abs_wet_in: float = 2.0,
                           abs_snow_in: float = 2.0) -> pd.DataFrame:
    """
    Flag daily extremes and simple streaks using °F and inches.
    Merges monthly normals via explicit MONTH key.
    """
    d = daily.copy()
    d["MONTH"] = d["DATE"].dt.month
    j = d.merge(normals, on=["STATION", "MONTH"], how="left")

    # Daily flags (relative to monthly normals and absolute cutoffs)
    j["IS_HOT_DAY"]   = (j["TMAX"] > (j["TMAX_NORMAL"] + rel_temp_thresh_f)) | (j["TMAX"] >= abs_hot_f)
    j["IS_COLD_DAY"]  = (j["TMIN"] < (j["TMIN_NORMAL"] - rel_temp_thresh_f)) | (j["TMIN"] <= abs_cold_f)
    j["IS_WET_DAY"]   = (j["PRCP"] >= abs_wet_in)
    j["IS_SNOWY_DAY"] = (j["SNOW"] >= abs_snow_in)

    # Streaks: 3+ consecutive hot or cold days, 2+ wet days
    j = j.sort_values(["STATION", "DATE"])
    def streak(series: pd.Series) -> pd.Series:
        # counts consecutive True values
        return series.groupby((~series).cumsum()).cumcount() + 1

    j["HOT_STREAK"]  = j.groupby("STATION")["IS_HOT_DAY"].transform(streak).where(j["IS_HOT_DAY"], 0)
    j["COLD_STREAK"] = j.groupby("STATION")["IS_COLD_DAY"].transform(streak).where(j["IS_COLD_DAY"], 0)
    j["WET_STREAK"]  = j.groupby("STATION")["IS_WET_DAY"].transform(streak).where(j["IS_WET_DAY"], 0)

    j["HEAT_WAVE"] = j["HOT_STREAK"]  >= 3
    j["COLD_SNAP"] = j["COLD_STREAK"] >= 3
    j["WET_PERIOD"] = j["WET_STREAK"] >= 2

    j["YEAR"] = j["DATE"].dt.year

    # Monthly event counts
    summary = (j.groupby(["STATION", "YEAR", "MONTH"])
                 .agg(IS_HOT_DAY=("IS_HOT_DAY", "sum"),
                      IS_COLD_DAY=("IS_COLD_DAY", "sum"),
                      IS_WET_DAY=("IS_WET_DAY", "sum"),
                      IS_SNOWY_DAY=("IS_SNOWY_DAY", "sum"),
                      HEAT_WAVE=("HEAT_WAVE", "sum"),
                      COLD_SNAP=("COLD_SNAP", "sum"),
                      WET_PERIOD=("WET_PERIOD", "sum"),
                      N_DAYS=("DATE", "nunique"))
                 .reset_index())
    # Prefix counts for clarity
    summary = summary.rename(columns={
        "IS_HOT_DAY": "N_HOT_DAYS",
        "IS_COLD_DAY": "N_COLD_DAYS",
        "IS_WET_DAY": "N_WET_DAYS",
        "IS_SNOWY_DAY": "N_SNOWY_DAYS"
    })
    return summary


# ----------------------------
# Trends (scipy linregress with graceful fallback)
# ----------------------------

def compute_trends(deviations: pd.DataFrame, period_type: str = "monthly") -> pd.DataFrame:
    """
    Compute linear trend of DEV_TAVG over years per (station, month) for monthly,
    or per station for weekly. Uses scipy.stats.linregress if available; otherwise
    returns slope only (R2 and p-value as NaN).
    """
    try:
        from scipy.stats import linregress
    except Exception:
        linregress = None

    dev = deviations.copy()
    if period_type == "monthly":
        group_cols = ["STATION", "MONTH"]
    else:
        group_cols = ["STATION"]

    out = []
    for key, g in dev.groupby(group_cols):
        # guard
        if "DEV_TAVG" not in g.columns:
            continue
        g = g.dropna(subset=["DEV_TAVG", "YEAR"])
        if g["YEAR"].nunique() < 2 or len(g) < 3:
            continue
        x = g["YEAR"].to_numpy()
        y = g["DEV_TAVG"].to_numpy()

        if linregress is not None:
            res = linregress(x, y)
            row = {
                "STATION": key[0] if isinstance(key, tuple) else key,
                "TREND_SLOPE": res.slope,
                "TREND_R2": res.rvalue ** 2,
                "TREND_PVAL": res.pvalue,
            }
            if period_type == "monthly":
                row["MONTH"] = key[1]
        else:
            # fallback: slope only via polyfit
            slope = np.polyfit(x, y, 1)[0]
            row = {
                "STATION": key[0] if isinstance(key, tuple) else key,
                "TREND_SLOPE": slope,
                "TREND_R2": np.nan,
                "TREND_PVAL": np.nan,
            }
            if period_type == "monthly":
                row["MONTH"] = key[1]

        out.append(row)

    return pd.DataFrame(out)


# ----------------------------
# Demo run in this notebook (if files exist)
# ----------------------------

DAILY_PATH = "/datadirectory/data/4109539.txt"
NORMALS_PATH = "/datadirectory/data/normals-monthly-2006-2020-2025-08-31T23-21-51.csv"
OUTDIR = "/datadirectory/data/outputs"

# Run if both input files are present
if os.path.exists(DAILY_PATH) and os.path.exists(NORMALS_PATH):
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)

    daily_df = read_daily_fixed_width(DAILY_PATH)
    normals_df = read_normals_csv(NORMALS_PATH)

    monthly_df = aggregate_monthly(daily_df)
    monthly_dev = compute_deviations(monthly_df, normals_df, "monthly")
    weekly_df = aggregate_weekly(daily_df)
    weekly_dev = compute_deviations(weekly_df, normals_df, "weekly")
    triple3_df = aggregate_rolling(daily_df, window=3)
    events_df = detect_critical_events(daily_df, normals_df)
    trends_df = compute_trends(monthly_dev, "monthly")

    # Save CSVs
    monthly_dev.to_csv(f"{OUTDIR}/monthly_deviations.csv", index=False)
    weekly_dev.to_csv(f"{OUTDIR}/weekly_deviations.csv", index=False)
    triple3_df.to_csv(f"{OUTDIR}/triple_day_summaries.csv", index=False)
    events_df.to_csv(f"{OUTDIR}/critical_events.csv", index=False)
    trends_df.to_csv(f"{OUTDIR}/monthly_trends.csv", index=False)

    # Show a quick preview to the user
    from caas_jupyter_tools import display_dataframe_to_user
    display_dataframe_to_user("Monthly deviations (first 24 rows)", monthly_dev.head(24))
    display_dataframe_to_user("Weekly deviations (first 24 rows)", weekly_dev.head(24))
    display_dataframe_to_user("Critical events (first 24 rows)", events_df.head(24))

    print(f"✔️ Outputs written to {OUTDIR}")
else:
    print("ℹ️ Place your input files at the paths below (or edit DAILY_PATH/NORMALS_PATH above) and re-run this cell:")
    print(f"  DAILY_PATH   = {DAILY_PATH}")
    print(f"  NORMALS_PATH = {NORMALS_PATH}")
    print(f"  OUTDIR       = {OUTDIR}")
