
from __future__ import annotations

from datetime import datetime, timedelta, timezone

# ----------------------------------------------------------------------------- 
# Determinism
# -----------------------------------------------------------------------------
SEED = 20260120

# -----------------------------------------------------------------------------
# Time frame
# -----------------------------------------------------------------------------
def _parse_iso(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)

DETECTION_TIME_DEFAULT: datetime = _parse_iso("2026-01-20T00:00:00Z")

METEO_HISTORY_HOURS = 48          # synthetic metocean covers t_now-48h .. t_now
METEO_TIME_STEP_MIN = 60          # hourly forcing
HINDCAST_HOURS = 36               # backward run depth (within the 36-48h spec band)
ORIGIN_TIME_WINDOW_H = 1.5        # +/- window around t_origin for correlation

# Reference instant for numeric time axis (seconds since this epoch)
EPOCH_REF: datetime = _parse_iso("2026-01-01T00:00:00Z")

# -----------------------------------------------------------------------------
# Spatial domain (Arabian Sea, west of Mumbai)
# -----------------------------------------------------------------------------
LAT_MIN, LAT_MAX = 18.0, 20.0
LON_MIN, LON_MAX = 71.0, 73.0
GRID_STEP_DEG = 0.1               # 21 x 21 nodes

KM_PER_DEG_LAT = 111.195          # equirectangular local scale
AVG_LAT = 0.5 * (LAT_MIN + LAT_MAX)
KM_PER_DEG_LON = KM_PER_DEG_LAT * __import__("math").cos(
    __import__("math").radians(AVG_LAT)
)  # ~105.1 km/deg at 19 N

# -----------------------------------------------------------------------------
# Slick (default request payload)
# -----------------------------------------------------------------------------
SLICK_LAT = 18.82
SLICK_LON = 72.45

# Anchor point where the target vessel crosses the hindcasted origin
ANCHOR_LAT = 19.35 - 0.40   # 18.95
ANCHOR_LON = 71.80 + 0.40   # 72.20

# -----------------------------------------------------------------------------
# Lagrangian particles
# -----------------------------------------------------------------------------
N_PARTICLES = 500
SEED_RADIUS_M = 1000.0
DRIFT_TIME_STEP_S = -900          # negative -> backwards in time (both engines)
WIND_DRIFT_FACTOR = 0.03          # 3 % wind leeway on top of surface currents
HORIZ_DIFFUSIVITY_M2_S = 3.0      # small sub-grid diffusion for ensemble spread

# -----------------------------------------------------------------------------
# Synthetic metocean field shape
# -----------------------------------------------------------------------------
CURRENT_UNIFORM_GUESS = (-0.03, -0.13)      # (u, v) m/s, refined by calibration
EDDY_CENTER = (72.55, 18.50)                # (lon, lat) mesoscale cold-core eddy
EDDY_RADIUS_KM = 40.0
EDDY_SPEED_MS = 0.30
TIDE_AMPLITUDE_UV = (0.06, 0.045)           # M2-like oscillation (12.42 h)
TIDE_PERIOD_S = 44712.0
WIND_U_MEAN_MS = 7.2                        # SW-monsoon 10 m wind, ERA5-like
WIND_V_MEAN_MS = 0.8

# -----------------------------------------------------------------------------
# Anomaly scoring (0..100)
# -----------------------------------------------------------------------------
SCORE_AIS_BLACKOUT = 40           # transponder gap > AIS_GAP_MIN_H near origin
SCORE_SPEED_DROP = 35             # SOG drop >= SPEED_DROP_FRACTION of cruise
SCORE_PROXIMITY = 25              # within PROXIMITY_KM of origin centroid
AIS_GAP_MIN_H = 2.0
SPEED_DROP_KEEP_FRACTION = 0.40   # min SOG <= 40 % of cruise  (>=60 % drop)
PROXIMITY_KM = 2.0
ANOMALY_LOOKBACK_H = 3.0          # scoring scans t_origin +/- this window
CANDIDATE_WINDOW_H = 1.5          # ship must cross polygon inside +/- window

# -----------------------------------------------------------------------------
# Target vessel (the "guilty" synthetic ship)
# -----------------------------------------------------------------------------
TARGET_MMSI = "419001234"
TARGET_NAME = "MT OCEAN TITAN"
TARGET_FLAG = "India"
TARGET_TYPE = "Crude Oil Tanker"
TARGET_CRUISE_KN = 14.0
TARGET_SLOW_KN = 3.0
TARGET_RESUME_KN = 13.5
TARGET_COURSE_DEG = 42.0          # deg true (SW -> NE transit)
GAP_START_H_BEFORE_ORIGIN = 2.5   # AIS blackout  [t_origin-2.5h, t_origin+1.0h]
GAP_END_H_AFTER_ORIGIN = 1.0      #   => 3.5 h total, centred on the rendezvous

# -----------------------------------------------------------------------------
# Engine selection
# -----------------------------------------------------------------------------
ENGINE_PREFERENCE = ("opendrift", "rk4")   # auto: first available wins
ENGINE_ENV_VAR = "AEGISX_ENGINE"           # override: "opendrift" | "rk4"

# -----------------------------------------------------------------------------
# Files
# -----------------------------------------------------------------------------
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
METEO_NC = DATA_DIR / "metocean_arabian_sea.nc"
AIS_SQLITE = DATA_DIR / "ais_tracks.sqlite"

# -----------------------------------------------------------------------------
# Convenience helpers
# -----------------------------------------------------------------------------
def detection_to_origin(t_detection: datetime) -> datetime:
    """Hindcasted origin time for a given slick detection time."""
    return t_detection - timedelta(hours=HINDCAST_HOURS)


def to_epoch_seconds(t: datetime) -> float:
    return (t - EPOCH_REF).total_seconds()


def from_epoch_seconds(s: float) -> datetime:
    return EPOCH_REF + timedelta(seconds=float(s))
