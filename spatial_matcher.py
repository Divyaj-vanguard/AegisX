"""
Synthetic AIS traffic and the spatio-temporal matcher for AegisX.

Responsibilities
----------------
1. Generate a deterministic synthetic AIS picture (stdlib sqlite3 backend at
   config.AIS_SQLITE) containing:
     * the target vessel  - MT OCEAN TITAN (config.TARGET_MMSI) which physically
       steams through the scenario anchor point exactly at t_origin, slows to
       3 kn 2.5 h beforehand, goes AIS-dark for 3.5 h and resumes transit;
     * six decoy ships steaming steady courses (one passes inside the
       uncertainty ellipse with zero anomalies, so partial scores are
       demonstrated).
2. match_candidates() - correlate every ship against a physics_engine
   HindcastResult (origin centroid, 95 % ellipse, time window) and compute the
   anomaly score defined in config:
       AIS blackout >= AIS_GAP_MIN_H      -> +SCORE_AIS_BLACKOUT
       speed drop to <=40 % of cruise     -> +SCORE_SPEED_DROP
       within PROXIMITY_KM of the origin  -> +SCORE_PROXIMITY
"""

from __future__ import annotations

import math
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from shapely.geometry import Point

import config
from physics_engine import HindcastResult

KN_TO_KMH = 1.852
TRACK_HALF_SPAN_H = 12.0          # ships sail  [t_origin - 12 h, t_origin + 12 h]
AIS_FIX_PERIOD_MIN = 10           # message interval
RESUME_AT_H = 3.0                 # target resumes 13.5 kn at t_origin + 3 h

# -----------------------------------------------------------------------------
# Synthetic fleet (metadata + trajectory definitions)
# -----------------------------------------------------------------------------

VESSEL_META: dict[str, dict] = {
    config.TARGET_MMSI: {
        "name": config.TARGET_NAME, "flag": config.TARGET_FLAG,
        "ship_type": config.TARGET_TYPE},
    "419555001": {"name": "MV CORAL VENTURE", "flag": "India",
                  "ship_type": "General Cargo"},
    "419000321": {"name": "FS SAGAR KIRTI", "flag": "India",
                  "ship_type": "Fishing"},
    "352004477": {"name": "MV ARABIAN STAR", "flag": "Panama",
                  "ship_type": "Bulk Carrier"},
    "538071234": {"name": "MT SULPHUR PRIDE", "flag": "Marshall Islands",
                  "ship_type": "Oil/Chemical Tanker"},
    "419001999": {"name": "MV KONKAN TRADER", "flag": "India",
                  "ship_type": "Container Ship"},
    "636092188": {"name": "ST OCEANIC HARMONY", "flag": "Liberia",
                  "ship_type": "Crude Oil Tanker"},
}


# -----------------------------------------------------------------------------
# Small geodesy helpers (equirectangular, consistent with config)
# -----------------------------------------------------------------------------

def _km_to_latlon(lat0: float, lon0: float, d_east_km: float, d_north_km: float):
    lat = lat0 + d_north_km / config.KM_PER_DEG_LAT
    lon = lon0 + d_east_km / config.KM_PER_DEG_LON
    return lat, lon


def dist_km(lat1, lon1, lat2, lon2):
    return math.hypot((lat1 - lat2) * config.KM_PER_DEG_LAT,
                      (lon1 - lon2) * config.KM_PER_DEG_LON)


def _point_segment_km(p, a, b):
    """
    Min distance (km) from point p to segment a-b plus the position along
    the segment (s in [0,1]) where the minimum is reached. Equirectangular.
    Inputs are (lat, lon) tuples.
    """
    px = (p[1],)  # silence linters on tuple element order
    ax = ((a[0] - p[0]) * config.KM_PER_DEG_LAT,
          (a[1] - p[1]) * config.KM_PER_DEG_LON)
    bx = ((b[0] - p[0]) * config.KM_PER_DEG_LAT,
          (b[1] - p[1]) * config.KM_PER_DEG_LON)
    dx, dy = bx[0] - ax[0], bx[1] - ax[1]
    L2 = dx * dx + dy * dy
    s = 0.0 if L2 == 0 else float(np.clip(-(ax[0] * dx + ax[1] * dy) / L2, 0.0, 1.0))
    cx, cy = ax[0] + s * dx, ax[1] + s * dy
    return math.hypot(cx, cy), s


def _course_components(course_deg: float) -> tuple[float, float]:
    """Unit (east, north) components of a true course."""
    r = math.radians(course_deg)
    return math.sin(r), math.cos(r)


# -----------------------------------------------------------------------------
# Target vessel kinematics (see config.py docstring for the scenario)
# -----------------------------------------------------------------------------

def _target_speed_kn(t_h: np.ndarray) -> np.ndarray:
    """Piecewise-constant SOG profile, knots, t in hours relative to t_origin."""
    t_h = np.asarray(t_h, dtype=float)
    sog = np.full_like(t_h, config.TARGET_CRUISE_KN)
    slow = (t_h >= -config.GAP_START_H_BEFORE_ORIGIN) & (t_h <= RESUME_AT_H)
    sog[slow] = config.TARGET_SLOW_KN
    sog[t_h > RESUME_AT_H] = config.TARGET_RESUME_KN
    return sog


def _target_along_track_km(t_h: np.ndarray) -> np.ndarray:
    """
    Signed distance along course from the anchor point, km.
    d(0) = 0  =>  the vessel crosses the anchor exactly at t_origin.
    """
    t_h = np.asarray(t_h, dtype=float)
    v_cruise = config.TARGET_CRUISE_KN * KN_TO_KMH
    v_slow = config.TARGET_SLOW_KN * KN_TO_KMH
    v_resume = config.TARGET_RESUME_KN * KN_TO_KMH
    t_slow = -config.GAP_START_H_BEFORE_ORIGIN

    d = np.empty_like(t_h)
    before = t_h <= t_slow
    middle = (~before) & (t_h <= RESUME_AT_H)
    after = t_h > RESUME_AT_H
    d[before] = v_slow * t_slow + v_cruise * (t_h[before] - t_slow)
    d[middle] = v_slow * t_h[middle]
    d[after] = v_slow * RESUME_AT_H + v_resume * (t_h[after] - RESUME_AT_H)
    return d


def _target_fixes(times_h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    d_km = _target_along_track_km(times_h)
    ue, un = _course_components(config.TARGET_COURSE_DEG)
    lat, lon = _km_to_latlon(config.ANCHOR_LAT, config.ANCHOR_LON,
                             d_km * ue, d_km * un)
    return lat, lon, _target_speed_kn(times_h)


# -----------------------------------------------------------------------------
# Decoy definitions: (mmsi, ref offset from anchor (east km, north km),
#                     t_ref relative to t_origin [h], course [deg T], speed [kn])
# CORAL VENTURE deliberately passes ~1.5 km from the origin inside the
# candidate window -> proximity score only. Everyone else stays far away.
# -----------------------------------------------------------------------------

def _normal_to(course_deg: float) -> tuple[float, float]:
    ue, un = _course_components(course_deg)
    return -un, ue   # 90 deg to port


_anchor_ne, _anchor_nn = _normal_to(300.0)
_DECOYS = [
    # mmsi            offset east km             offset north km             t_ref_h  course  speed
    ("419555001", 1.5 * _anchor_ne, 1.5 * _anchor_nn, -0.3, 300.0, 11.0),
    ("419000321", 25.0, -4.0, 2.0, 80.0, 3.8),
    ("352004477", 8.0, 20.0, 0.5, 295.0, 12.5),
    ("538071234", 22.0, -21.0, -1.0, 115.0, 13.0),
    ("419001999", 3.0, -12.0, 1.0, 260.0, 15.0),
    ("636092188", -28.0, 35.0, -2.0, 50.0, 14.0),
]


def _steady_fixes(e_ref_km: float, n_ref_km: float, t_ref_h: float,
                  course_deg: float, speed_kn: float,
                  times_h: np.ndarray):
    """Position of a ship passing the reference offset at t_ref_h."""
    ue, un = _course_components(course_deg)
    d_km = speed_kn * KN_TO_KMH * (times_h - t_ref_h)
    lat, lon = _km_to_latlon(config.ANCHOR_LAT, config.ANCHOR_LON,
                             e_ref_km + d_km * ue, n_ref_km + d_km * un)
    sog = np.full_like(times_h, speed_kn)
    return lat, lon, sog


# -----------------------------------------------------------------------------
# Dataset generation -> pandas DataFrame -> sqlite
# -----------------------------------------------------------------------------

def generate_ais_dataframe(t_origin: datetime | None = None) -> pd.DataFrame:
    """Build the full synthetic AIS picture as a long-form DataFrame."""
    t_origin = t_origin or config.detection_to_origin(config.DETECTION_TIME_DEFAULT)
    n_fixes = int(2 * TRACK_HALF_SPAN_H * 60 / AIS_FIX_PERIOD_MIN) + 1
    times_h = np.linspace(-TRACK_HALF_SPAN_H, TRACK_HALF_SPAN_H, n_fixes)

    rows: list[dict] = []

    def emit(mmsi: str, lat, lon, sog, cog, mask=None):
        ts = [t_origin + timedelta(hours=float(h)) for h in times_h]
        for i in range(len(times_h)):
            if mask is not None and not mask[i]:
                continue                       # AIS blackout: no report
            rows.append({
                "mmsi": mmsi, "ts": ts[i],
                "lat": round(float(lat[i]), 6), "lon": round(float(lon[i]), 6),
                "sog": round(float(sog[i]), 2), "cog": round(float(cog), 1),
            })

    # -- target vessel with blackout -----------------------------------------
    lat, lon, sog = _target_fixes(times_h)
    blackout = (times_h > -config.GAP_START_H_BEFORE_ORIGIN) & \
               (times_h < config.GAP_END_H_AFTER_ORIGIN)
    emit(config.TARGET_MMSI, lat, lon, sog, config.TARGET_COURSE_DEG,
         mask=~blackout)

    # -- decoys ---------------------------------------------------------------
    for mmsi, e_km, n_km, t_ref, course, speed in _DECOYS:
        lat, lon, sog = _steady_fixes(e_km, n_km, t_ref, course, speed, times_h)
        emit(mmsi, lat, lon, sog, course)

    df = pd.DataFrame(rows).sort_values(["mmsi", "ts"]).reset_index(drop=True)
    return df


def write_ais_sqlite(df: pd.DataFrame | None = None,
                     path: Path | None = None) -> Path:
    """Persist the synthetic AIS picture (tables: ais, vessels)."""
    path = Path(path or config.AIS_SQLITE)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_ais_dataframe() if df is None else df
    tmp = df.copy()
    tmp["ts"] = pd.to_datetime(tmp["ts"], utc=True).dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    if path.exists():
        path.unlink()
    con = sqlite3.connect(path)
    try:
        cur = con.cursor()
        cur.execute("""CREATE TABLE vessels(
                        mmsi TEXT PRIMARY KEY, name TEXT, flag TEXT,
                        ship_type TEXT)""")
        cur.execute("""CREATE TABLE ais(
                        mmsi TEXT, ts TEXT, lat REAL, lon REAL,
                        sog REAL, cog REAL)""")
        cur.execute("CREATE INDEX idx_ais_mmsi_ts ON ais(mmsi, ts)")
        cur.executemany(
            "INSERT INTO vessels VALUES (?,?,?,?)",
            [(m, v["name"], v["flag"], v["ship_type"])
             for m, v in VESSEL_META.items()])
        cur.executemany(
            "INSERT INTO ais VALUES (?,?,?,?,?,?)",
            list(tmp[["mmsi", "ts", "lat", "lon", "sog", "cog"]]
                 .itertuples(index=False, name=None)))
        con.commit()
    finally:
        con.close()
    return path


def ensure_ais_db(path: Path | None = None) -> Path:
    """Create the AIS database if missing, otherwise reuse it."""
    path = Path(path or config.AIS_SQLITE)
    if not path.exists():
        write_ais_sqlite(path=path)
        return path
    try:                                   # sanity-check existing file
        con = sqlite3.connect(path)
        n = con.execute("SELECT COUNT(*) FROM ais").fetchone()[0]
        con.close()
        if n > 0:
            return path
    except sqlite3.Error:
        pass
    return write_ais_sqlite(path=path)


def load_ais(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (tracks, vessels). tracks.ts is tz-aware UTC."""
    path = ensure_ais_db(path)
    con = sqlite3.connect(path)
    try:
        tracks = pd.read_sql_query("SELECT * FROM ais ORDER BY mmsi, ts", con)
        vessels = pd.read_sql_query("SELECT * FROM vessels", con)
    finally:
        con.close()
    tracks["ts"] = pd.to_datetime(tracks["ts"], utc=True, format="ISO8601")
    return tracks, vessels


# -----------------------------------------------------------------------------
# Matching + anomaly scoring
# -----------------------------------------------------------------------------

@dataclass
class VesselReport:
    mmsi: str
    name: str
    flag: str
    ship_type: str
    n_fixes: int
    is_candidate: bool          # fix inside origin ellipse within the window
    closest_km: float
    closest_time: datetime | None
    max_gap_h: float
    gap_start: datetime | None
    gap_end: datetime | None
    gap_exit_lat: float | None  # first fix after the gap (for map drawing)
    gap_exit_lon: float | None
    gap_entry_lat: float | None
    gap_entry_lon: float | None
    min_sog_kn: float
    cruise_sog_kn: float
    flags: list[str]
    score: int
    verdict: str
    track: pd.DataFrame = field(repr=False)


def _verdict(score: int) -> str:
    if score >= 70:
        return "PRIME SUSPECT"
    if score >= 40:
        return "SUSPICIOUS"
    if score >= config.SCORE_PROXIMITY:
        return "POSSIBLE WITNESS"
    return "CLEAR"


def match_candidates(hindcast: HindcastResult,
                     tracks: pd.DataFrame | None = None,
                     vessels: pd.DataFrame | None = None) -> list[VesselReport]:
    """
    Correlate every AIS track with the hindcast origin and score anomalies.
    Returns reports sorted by score (desc) then closest approach (asc).
    """
    if tracks is None or vessels is None:
        tracks, vessels = load_ais()
    meta = {r["mmsi"]: r for _, r in vessels.iterrows()}

    t0 = _as_utc(hindcast.t_origin)
    w = timedelta(hours=config.CANDIDATE_WINDOW_H)
    lb = timedelta(hours=config.ANOMALY_LOOKBACK_H)
    poly = hindcast.detect_polygon

    reports: list[VesselReport] = []
    for mmsi, g in tracks.groupby("mmsi"):
        g = g.sort_values("ts").reset_index(drop=True)

        # -- spatial / temporal window: fixes near t_origin -------------------
        wg = g[(g["ts"] >= t0 - w) & (g["ts"] <= t0 + w)]
        in_poly = False
        if len(wg):
            inside = [poly.covers(Point(ln, lt))
                      for ln, lt in zip(wg["lon"], wg["lat"])]
            in_poly = any(inside)

        # -- closest approach to the origin centroid --------------------------
        dwg = wg if len(wg) else g[(g["ts"] >= t0 - lb) & (g["ts"] <= t0 + lb)]
        if len(dwg):
            dists = np.hypot(
                (dwg["lat"].to_numpy() - hindcast.origin_lat) * config.KM_PER_DEG_LAT,
                (dwg["lon"].to_numpy() - hindcast.origin_lon) * config.KM_PER_DEG_LON)
            i_min = int(np.argmin(dists))
            closest_km = float(dists[i_min])
            closest_time = dwg["ts"].iloc[i_min].to_pydatetime()
        else:
            closest_km, closest_time = float("inf"), None

        # -- AIS blackout scan (gaps intersecting the lookback window) --------
        max_gap_h = 0.0
        gap_start = gap_end = None
        gap_exit = gap_entry = None
        dts = g["ts"].diff().dt.total_seconds().to_numpy()
        for i in range(1, len(g)):
            gap_s = dts[i]
            if gap_s < config.AIS_GAP_MIN_H * 3600:
                continue
            a, b = g["ts"].iloc[i - 1], g["ts"].iloc[i]
            if b < t0 - lb or a > t0 + lb:          # gap outside lookback
                continue
            if gap_s / 3600.0 > max_gap_h:
                max_gap_h = gap_s / 3600.0
                gap_start, gap_end = a.to_pydatetime(), b.to_pydatetime()
                gap_entry = (float(g["lat"].iloc[i - 1]), float(g["lon"].iloc[i - 1]))
                gap_exit = (float(g["lat"].iloc[i]), float(g["lon"].iloc[i]))

        # -- speed profile within lookback ------------------------------------
        lg = g[(g["ts"] >= t0 - lb) & (g["ts"] <= t0 + lb)]
        min_sog = float(lg["sog"].min()) if len(lg) else float("nan")
        cruise_sog = float(g["sog"].max())

        # -- scoring -----------------------------------------------------------
        flags: list[str] = []
        score = 0
        if in_poly and math.isfinite(closest_km) and closest_km <= config.PROXIMITY_KM:
            score += config.SCORE_PROXIMITY
            flags.append(f"within {closest_km:.2f} km of hindcast origin "
                         f"(<= {config.PROXIMITY_KM} km)")
        if max_gap_h >= config.AIS_GAP_MIN_H:
            score += config.SCORE_AIS_BLACKOUT
            flags.append(f"AIS blackout {max_gap_h:.1f} h "
                         f"({gap_start:%H:%M}Z -> {gap_end:%H:%M}Z)")
        if (math.isfinite(min_sog) and cruise_sog > 0
                and min_sog <= config.SPEED_DROP_KEEP_FRACTION * cruise_sog):
            score += config.SCORE_SPEED_DROP
            flags.append(f"speed drop to {min_sog:.1f} kn "
                         f"({100 * min_sog / cruise_sog:.0f}% of {cruise_sog:.1f} kn cruise)")

        m = meta.get(mmsi, {})
        reports.append(VesselReport(
            mmsi=mmsi,
            name=str(m.get("name", "UNKNOWN")),
            flag=str(m.get("flag", "?")),
            ship_type=str(m.get("ship_type", "?")),
            n_fixes=len(g),
            is_candidate=in_poly,
            closest_km=closest_km,
            closest_time=closest_time,
            max_gap_h=max_gap_h,
            gap_start=gap_start, gap_end=gap_end,
            gap_exit_lat=gap_exit[0] if gap_exit else None,
            gap_exit_lon=gap_exit[1] if gap_exit else None,
            gap_entry_lat=gap_entry[0] if gap_entry else None,
            gap_entry_lon=gap_entry[1] if gap_entry else None,
            min_sog_kn=min_sog, cruise_sog_kn=cruise_sog,
            flags=flags, score=score, verdict=_verdict(score),
            track=g,          # full +/-12 h track (window slice computed by caller)
        ))

    reports.sort(key=lambda r: (-r.score, r.closest_km))
    return reports


def top_suspects(reports: list[VesselReport]) -> list[VesselReport]:
    """Candidates that actually entered the origin ellipse, ranked."""
    return [r for r in reports if r.is_candidate]


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def _as_utc(t: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(t)
    return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")


if __name__ == "__main__":
    path = write_ais_sqlite()
    tr, vv = load_ais()
    print(f"AIS database: {path}  ({len(tr)} fixes, {tr['mmsi'].nunique()} vessels)")
    print(tr.groupby("mmsi").agg(first=("ts", "first"), last=("ts", "last"),
                                 fixes=("ts", "size"), sog_min=("sog", "min"),
                                 sog_max=("sog", "max")))
