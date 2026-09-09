"""
Lagrangian hindcast physics engine for AegisX.

Primary engine: OpenDrift (OceanDrift) running backwards with a negative
time step over the synthetic NetCDF metocean.

Fallback engine: self-contained vectorised Runge-Kutta 4 backtracker that
samples the same NetCDF (or, if even that is missing, the analytic field),
so the pipeline degrades gracefully instead of failing.

Both engines return a common HindcastResult with the particle endpoints,
mean back-trajectory, origin estimate and a 95 % uncertainty ellipse.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta

import numpy as np
from shapely.geometry import Point, Polygon

import config
import metfield

# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class HindcastResult:
    engine: str                          # "opendrift" | "rk4-netcdf" | "rk4-analytic"
    slick_lon: float
    slick_lat: float
    detection_time: datetime
    t_origin: datetime                   # center of origin window
    window_start: datetime
    window_end: datetime
    origin_lon: float                    # mean endpoint
    origin_lat: float
    endpoints_lon: np.ndarray            # (M,) surviving particles
    endpoints_lat: np.ndarray
    path_lon: np.ndarray                 # mean trajectory, detection -> origin
    path_lat: np.ndarray
    path_times: list                     # list[datetime], same length as path
    ellipse_lonlat: list                 # [(lon, lat), ...] closed ring (95 %)
    ellipse: dict                        # center / semi axes (km) / rotation
    detect_polygon: object = field(default=None, repr=False)   # shapely polygon
    n_particles: int = 0

    def centroid(self) -> tuple[float, float]:
        return self.origin_lat, self.origin_lon


# ---------------------------------------------------------------------------
# Uncertainty ellipse (covariance -> 95 % ellipse polygon in lon/lat)
# ---------------------------------------------------------------------------

def covariance_ellipse(lon: np.ndarray, lat: np.ndarray,
                       scale: float = 2.4477, min_semi_km: float = 3.5,
                       n_pts: int = 72):
    """
    95 % confidence ellipse of the endpoint cloud (scale^2 = chi2_2(0.95)).
    Semi-axes are clamped to >= min_semi_km so the search polygon stays
    operationally meaningful even for tight ensembles.
    Returns (polygon_lonlat, meta_dict).
    """
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    clon, clat = float(lon.mean()), float(lat.mean())

    km_y = (lat - clat) * config.KM_PER_DEG_LAT
    km_x = (lon - clon) * config.KM_PER_DEG_LON
    X = np.vstack([km_x, km_y])
    cov = np.cov(X) if X.shape[1] > 3 else np.eye(2) * 1e-4
    vals, vecs = np.linalg.eigh(cov)
    vals = np.clip(vals, 0.0, None)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]

    semi = scale * np.sqrt(vals)                       # km
    semi = np.maximum(semi, min_semi_km)
    theta = math.atan2(vecs[1, 0], vecs[0, 0])         # rotation of major axis

    a = np.linspace(0, 2 * np.pi, n_pts)
    ex = semi[0] * np.cos(a)
    ey = semi[1] * np.sin(a)
    rx = ex * math.cos(theta) - ey * math.sin(theta)
    ry = ex * math.sin(theta) + ey * math.cos(theta)
    ring_lon = clon + rx / config.KM_PER_DEG_LON
    ring_lat = clat + ry / config.KM_PER_DEG_LAT
    ring = list(zip(np.round(ring_lon, 6), np.round(ring_lat, 6)))
    ring.append(ring[0])

    meta = {
        "center_lon": round(clon, 6),
        "center_lat": round(clat, 6),
        "semi_major_km": round(float(semi[0]), 3),
        "semi_minor_km": round(float(semi[1]), 3),
        "rotation_deg": round(math.degrees(theta), 2),
        "confidence": 0.95,
    }
    return ring, meta


# ---------------------------------------------------------------------------
# Particle seeding (shared)
# ---------------------------------------------------------------------------

def seed_particles(slick_lon: float, slick_lat: float, n: int, radius_m: float):
    """Gaussian seeding of n particles around the slick centroid."""
    rng = np.random.default_rng(config.SEED)
    r_m = np.abs(rng.normal(0.0, radius_m / 2.0, n))
    bearing = rng.uniform(0.0, 2 * np.pi, n)
    lat = slick_lat + (r_m * np.cos(bearing)) / (config.KM_PER_DEG_LAT * 1000.0)
    lon = slick_lon + (r_m * np.sin(bearing)) / (
        config.KM_PER_DEG_LAT * 1000.0 * math.cos(math.radians(slick_lat)))
    return lon, lat


# ---------------------------------------------------------------------------
# Engine 1: built-in RK4 backtracker (fallback, zero C-dependencies needed)
# ---------------------------------------------------------------------------

def hindcast_rk4(slick_lat: float, slick_lon: float, detection_time: datetime,
                 prefer_netcdf: bool = True) -> HindcastResult:
    sampler_kind = "netcdf" if (prefer_netcdf and config.METEO_NC.exists()) else "analytic"
    sampler = metfield.get_sampler(prefer_netcdf=prefer_netcdf)

    lon0, lat0 = seed_particles(slick_lon, slick_lat, config.N_PARTICLES,
                                config.SEED_RADIUS_M)
    t0 = config.to_epoch_seconds(detection_time)
    rng = np.random.default_rng(config.SEED + 7)

    traj, times = metfield.integrate_rk4(
        sampler.eff_vel, lon0, lat0, t0,
        -config.HINDCAST_HOURS * 3600.0,
        config.DRIFT_TIME_STEP_S,
        rng=rng, diffusivity_m2s=config.HORIZ_DIFFUSIVITY_M2_S,
    )

    end_lon = traj[-1, :, 0]
    end_lat = traj[-1, :, 1]
    path_lon = traj[:, :, 0].mean(axis=1)
    path_lat = traj[:, :, 1].mean(axis=1)
    path_times = [config.from_epoch_seconds(s) for s in times]

    return _pack_result(
        engine=f"rk4-{sampler_kind}", slick_lat=slick_lat, slick_lon=slick_lon,
        detection_time=detection_time, end_lon=end_lon, end_lat=end_lat,
        path_lon=path_lon, path_lat=path_lat, path_times=path_times)


# ---------------------------------------------------------------------------
# Engine 2: OpenDrift OceanDrift (primary when available)
# ---------------------------------------------------------------------------

def hindcast_opendrift(slick_lat: float, slick_lon: float,
                       detection_time: datetime) -> HindcastResult:
    if not config.METEO_NC.exists():
        raise RuntimeError("metocean NetCDF missing; cannot use OpenDrift engine")

    import logging
    import random

    # Deterministic seeding across OpenDrift's RNGs
    np.random.seed(config.SEED)
    random.seed(config.SEED)
    logging.getLogger("opendrift").setLevel(logging.CRITICAL)

    from opendrift.models.oceandrift import OceanDrift
    from opendrift.readers import reader_netCDF_CF_generic

    o = OceanDrift(loglevel=50)
    o.add_reader(reader_netCDF_CF_generic.Reader(str(config.METEO_NC)))
    o.set_config("general:coastline_action", "none")     # fully offshore domain
    o.set_config("drift:advection_scheme", "runge-kutta4")
    try:
        o.set_config("drift:vertical_mixing", False)
    except Exception:
        pass
    try:
        o.set_config("environment:fallback:x_sea_water_velocity", 0.0)
        o.set_config("environment:fallback:y_sea_water_velocity", 0.0)
    except Exception:
        pass

    t_det_naive = detection_time.replace(tzinfo=None)
    o.seed_elements(lon=slick_lon, lat=slick_lat, number=config.N_PARTICLES,
                    radius=config.SEED_RADIUS_M, time=t_det_naive,
                    wind_drift_factor=config.WIND_DRIFT_FACTOR)

    hours = config.HINDCAST_HOURS
    run_errors = []
    for kwargs in ({"duration": timedelta(hours=-hours), "time_step": config.DRIFT_TIME_STEP_S},
                   {"steps": int(hours * 3600 / abs(config.DRIFT_TIME_STEP_S)),
                    "time_step": config.DRIFT_TIME_STEP_S}):
        try:
            o.run(**kwargs)
            break
        except Exception as exc:                                       # pragma: no cover
            run_errors.append(f"{kwargs}: {exc}")
    else:
        raise RuntimeError("OpenDrift run failed: " + " | ".join(run_errors))

    # OpenDrift >= 1.11 exposes results as an xarray Dataset on .result
    # (dims: trajectory x time); older versions had the .history dict.
    if hasattr(o, "result") and "lon" in o.result:
        lon_h = np.asarray(o.result["lon"].values)   # (particles, steps)
        lat_h = np.asarray(o.result["lat"].values)
    else:                                            # pragma: no cover - old API
        lon_h = np.asarray(o.history["lon"])
        lat_h = np.asarray(o.history["lat"])
    if lon_h.ndim != 2:
        raise RuntimeError("unexpected OpenDrift history shape")

    # final column = oldest time reached
    end_lon = lon_h[:, -1].astype(float)
    end_lat = lat_h[:, -1].astype(float)
    ok = np.isfinite(end_lon) & np.isfinite(end_lat)
    if ok.sum() < config.N_PARTICLES * 0.5:
        raise RuntimeError(f"too many particles lost ({ok.sum()} of {len(end_lon)})")
    end_lon, end_lat = end_lon[ok], end_lat[ok]

    path_lon = np.nanmean(lon_h, axis=0)
    path_lat = np.nanmean(lat_h, axis=0)
    n_steps = lon_h.shape[1]
    path_times = [detection_time + timedelta(seconds=config.DRIFT_TIME_STEP_S * k)
                  for k in range(n_steps)]

    return _pack_result(
        engine="opendrift", slick_lat=slick_lat, slick_lon=slick_lon,
        detection_time=detection_time, end_lon=end_lon, end_lat=end_lat,
        path_lon=path_lon, path_lat=path_lat, path_times=path_times)


# ---------------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------------

def _pack_result(engine: str, slick_lat: float, slick_lon: float,
                 detection_time: datetime, end_lon: np.ndarray, end_lat: np.ndarray,
                 path_lon, path_lat, path_times) -> HindcastResult:
    t_origin = config.detection_to_origin(detection_time)
    w = timedelta(hours=config.ORIGIN_TIME_WINDOW_H)
    ring, meta = covariance_ellipse(end_lon, end_lat)
    poly = Polygon(ring)
    return HindcastResult(
        engine=engine,
        slick_lon=slick_lon, slick_lat=slick_lat,
        detection_time=detection_time,
        t_origin=t_origin, window_start=t_origin - w, window_end=t_origin + w,
        origin_lon=float(np.mean(end_lon)), origin_lat=float(np.mean(end_lat)),
        endpoints_lon=np.asarray(end_lon), endpoints_lat=np.asarray(end_lat),
        path_lon=np.asarray(path_lon), path_lat=np.asarray(path_lat),
        path_times=list(path_times),
        ellipse_lonlat=ring, ellipse=meta, detect_polygon=poly,
        n_particles=len(end_lon),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_hindcast(slick_lat: float, slick_lon: float, detection_time: datetime,
                 engine: str | None = None) -> HindcastResult:
    """
    Run the attribution hindcast. Engine order:
        explicit argument > AEGISX_ENGINE env var > config preference.
    Falls back through the chain on any failure, never raises for engine issues.
    """
    forced = engine or os.environ.get(config.ENGINE_ENV_VAR)
    order = ([forced] if forced else list(config.ENGINE_PREFERENCE))

    errors = []
    for eng in order:
        try:
            if eng == "opendrift":
                return hindcast_opendrift(slick_lat, slick_lon, detection_time)
            if eng == "rk4":
                return hindcast_rk4(slick_lat, slick_lon, detection_time)
        except Exception as exc:
            errors.append(f"{eng}: {exc}")
    # last resort: pure analytic RK4 (cannot fail unless code broken)
    res = hindcast_rk4(slick_lat, slick_lon, detection_time, prefer_netcdf=False)
    res.engine += " (fallback after: " + "; ".join(errors) + ")"
    return res


# ---------------------------------------------------------------------------
# CLI self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import metfield  # noqa: F401  (ensure generated file path OK)

    if not config.METEO_NC.exists():
        print("metocean NetCDF missing - generating first")
        metfield.write_netcdf()

    det = config.DETECTION_TIME_DEFAULT
    for eng in ("rk4", "opendrift"):
        try:
            r = run_hindcast(config.SLICK_LAT, config.SLICK_LON, det, engine=eng)
        except Exception as exc:
            print(f"[{eng}] FAILED: {exc}")
            continue
        err_km = math.hypot(
            (r.origin_lat - config.ANCHOR_LAT) * config.KM_PER_DEG_LAT,
            (r.origin_lon - config.ANCHOR_LON) * config.KM_PER_DEG_LON)
        print(f"[{eng}] engine={r.engine}  origin=({r.origin_lat:.5f}, {r.origin_lon:.5f})  "
              f"t_origin={r.t_origin.isoformat()}  anchor-error={err_km:.2f} km  "
              f"particles={r.n_particles}  ellipse "
              f"{r.ellipse['semi_major_km']}x{r.ellipse['semi_minor_km']} km")
