"""
Synthetic metocean field for AegisX.

The field is analytic and fully deterministic:

    currents = uniform flow (calibrated) + mesoscale cold-core eddy + M2 tide
    winds    = ERA5-like SW-monsoon 10 m wind (diurnal cycle + weak gradients)

The uniform current component is *calibrated* so that a backward drift from
the default slick centroid over HINDCAST_HOURS ends exactly on the scenario
anchor point (ANCHOR_LAT, ANCHOR_LON). This guarantees that the synthetic
target vessel, which physically steams through the anchor, matches the
hindcasted origin of the slick.

This module also owns:
  * integrate_rk4()  - vectorised Runge-Kutta 4 particle integrator
  * NetCDF writer    - CF-flavoured file consumable by OpenDrift
  * NetCDFSampler    - trilinear (bilinear in space, linear in time) sampler
  * AnalyticSampler  - pure-formula fallback when the NetCDF is missing
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np

import config

# ---------------------------------------------------------------------------
# Analytic field
# ---------------------------------------------------------------------------

def _eddy(lon: np.ndarray, lat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cyclonic (counter-clockwise) mesoscale eddy, m/s."""
    clon, clat = config.EDDY_CENTER
    dx_km = (lon - clon) * config.KM_PER_DEG_LON
    dy_km = (lat - clat) * config.KM_PER_DEG_LAT
    r_km = np.hypot(dx_km, dy_km)
    r_safe = np.maximum(r_km, 0.5)
    R = config.EDDY_RADIUS_KM
    # velocity profile peaks at r = R
    v_theta = config.EDDY_SPEED_MS * (r_km / R) * np.exp(0.5 * (1.0 - (r_km / R) ** 2))
    u = -v_theta * dy_km / r_safe
    v = v_theta * dx_km / r_safe
    return u, v


def _tide(t_s: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Spatially uniform M2-like tidal oscillation, m/s."""
    om = 2.0 * np.pi / config.TIDE_PERIOD_S
    au, av = config.TIDE_AMPLITUDE_UV
    return au * np.sin(om * t_s + 0.7), av * np.sin(om * t_s + 0.7 + 0.9)


def current_uv(lon, lat, t_s, uniform: tuple[float, float]) -> tuple[np.ndarray, np.ndarray]:
    """Surface current (eastward, northward) in m/s. Broadcastable inputs."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    ue, ve = _eddy(lon, lat)
    ut, vt = _tide(t_s)
    u = uniform[0] + ue + ut
    v = uniform[1] + ve + vt
    return u, v


def wind_uv(lon, lat, t_s) -> tuple[np.ndarray, np.ndarray]:
    """10 m wind (eastward, northward) in m/s, ERA5-like SW monsoon."""
    lon = np.asarray(lon, dtype=float)
    lat = np.asarray(lat, dtype=float)
    t_s = np.asarray(t_s, dtype=float)
    day = 2.0 * np.pi * t_s / 86400.0
    u10 = config.WIND_U_MEAN_MS + 1.6 * np.sin(day + 0.4) + 0.5 * (lat - 19.0)
    v10 = config.WIND_V_MEAN_MS + 1.1 * np.cos(day + 1.1) - 0.3 * (lon - 72.0)
    return u10, v10


# ---------------------------------------------------------------------------
# Vectorised RK4 integrator (shared by the physics engine and calibration)
# ---------------------------------------------------------------------------

def integrate_rk4(vel_fn, lon0, lat0, t0_s: float, duration_s: float, dt_s: float,
                  rng: np.random.Generator | None = None,
                  diffusivity_m2s: float = 0.0):
    """
    Integrate particles with classical RK4.

    vel_fn(lon, lat, t_s) -> (u, v) in m/s (effective velocity incl. windage).
    duration_s and dt_s negative => backward integration.
    Returns (traj, times): traj shape (nsteps+1, N, 2) holding (lon, lat).
    """
    lon = np.atleast_1d(np.asarray(lon0, dtype=float)).copy()
    lat = np.atleast_1d(np.asarray(lat0, dtype=float)).copy()
    n = lon.size
    nsteps = int(round(abs(duration_s / dt_s)))
    if nsteps < 1:
        raise ValueError("duration must be at least one time step")
    dt = float(dt_s)
    if np.sign(duration_s) != np.sign(dt):
        dt = -dt

    traj = np.empty((nsteps + 1, n, 2), dtype=np.float64)
    traj[0, :, 0] = lon
    traj[0, :, 1] = lat
    times = t0_s + dt * np.arange(nsteps + 1)

    deg_lat_m = config.KM_PER_DEG_LAT * 1000.0

    def step_velocity(lo, la, t):
        u, v = vel_fn(lo, la, t)
        deg_lon_m = config.KM_PER_DEG_LAT * 1000.0 * np.cos(np.radians(la))
        return u / deg_lon_m, v / deg_lat_m  # deg/s

    t = float(t0_s)
    sigma_m = math.sqrt(2.0 * diffusivity_m2s * abs(dt)) if diffusivity_m2s > 0 else 0.0

    for k in range(nsteps):
        k1u, k1v = step_velocity(lon, lat, t)
        k2u, k2v = step_velocity(lon + 0.5 * dt * k1u, lat + 0.5 * dt * k1v, t + 0.5 * dt)
        k3u, k3v = step_velocity(lon + 0.5 * dt * k2u, lat + 0.5 * dt * k2v, t + 0.5 * dt)
        k4u, k4v = step_velocity(lon + dt * k3u, lat + dt * k3v, t + dt)
        lon = lon + dt / 6.0 * (k1u + 2.0 * k2u + 2.0 * k3u + k4u)
        lat = lat + dt / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v)
        lat = np.clip(lat, -89.0, 89.0)

        if sigma_m > 0.0 and rng is not None:
            lat_m = rng.normal(0.0, sigma_m, n)
            lon_m = rng.normal(0.0, sigma_m, n)
            lat = lat + lat_m / deg_lat_m
            lon = lon + lon_m / (deg_lat_m * np.cos(np.radians(lat)))

        traj[k + 1, :, 0] = lon
        traj[k + 1, :, 1] = lat
        t += dt

    return traj, times


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------

class AnalyticSampler:
    """Pure-formula fallback field (no NetCDF required)."""

    def __init__(self, uniform: tuple[float, float]):
        self.uniform = uniform

    def current_wind(self, lon, lat, t_s):
        u, v = current_uv(lon, lat, t_s, self.uniform)
        u10, v10 = wind_uv(lon, lat, t_s)
        return u, v, u10, v10

    def eff_vel(self, lon, lat, t_s):
        u, v, u10, v10 = self.current_wind(lon, lat, t_s)
        w = config.WIND_DRIFT_FACTOR
        return u + w * u10, v + w * v10


class NetCDFSampler:
    """Bilinear-in-space, linear-in-time sampler over the generated NetCDF."""

    def __init__(self, path: Path):
        import netCDF4  # deferred so the analytic fallback has no C-dependency

        self._path = Path(path)
        with netCDF4.Dataset(self._path) as ds:
            self.lon = np.asarray(ds["lon"][:], dtype=np.float64)
            self.lat = np.asarray(ds["lat"][:], dtype=np.float64)
            self.t_s = np.asarray(ds["time"][:], dtype=np.float64)
            self.U = np.asarray(ds["u_current"][:], dtype=np.float64)
            self.V = np.asarray(ds["v_current"][:], dtype=np.float64)
            self.U10 = np.asarray(ds["u10"][:], dtype=np.float64)
            self.V10 = np.asarray(ds["v10"][:], dtype=np.float64)

    # -- low level gather ----------------------------------------------------
    def _interp(self, F, lon, lat, t_s):
        lon = np.atleast_1d(np.asarray(lon, dtype=float))
        lat = np.atleast_1d(np.asarray(lat, dtype=float))
        t_s = float(np.asarray(t_s).flat[0])

        ix = np.clip(np.searchsorted(self.lon, lon) - 1, 0, len(self.lon) - 2)
        jy = np.clip(np.searchsorted(self.lat, lat) - 1, 0, len(self.lat) - 2)
        kt = int(np.clip(np.searchsorted(self.t_s, t_s) - 1, 0, len(self.t_s) - 2))

        fx = (lon - self.lon[ix]) / (self.lon[ix + 1] - self.lon[ix])
        fy = (lat - self.lat[jy]) / (self.lat[jy + 1] - self.lat[jy])
        ft = (t_s - self.t_s[kt]) / (self.t_s[kt + 1] - self.t_s[kt])
        
        fx = np.clip(fx, 0.0, 1.0)
        fy = np.clip(fy, 0.0, 1.0)
        ft = np.clip(ft, 0.0, 1.0)

        # Explicitly take the 2D slices (ny, nx) at kt and kt+1
        F0 = F[kt]
        F1 = F[kt + 1]

        f00_0 = F0[jy, ix]; f10_0 = F0[jy, ix + 1]
        f01_0 = F0[jy + 1, ix]; f11_0 = F0[jy + 1, ix + 1]
        bil0 = (f00_0 * (1 - fx) + f10_0 * fx) * (1 - fy) + (f01_0 * (1 - fx) + f11_0 * fx) * fy

        f00_1 = F1[jy, ix]; f10_1 = F1[jy, ix + 1]
        f01_1 = F1[jy + 1, ix]; f11_1 = F1[jy + 1, ix + 1]
        bil1 = (f00_1 * (1 - fx) + f10_1 * fx) * (1 - fy) + (f01_1 * (1 - fx) + f11_1 * fx) * fy

        val = bil0 * (1 - ft) + bil1 * ft
        return val if val.size > 1 else float(val.flat[0])

    def current_wind(self, lon, lat, t_s):
        lon = np.asarray(lon, dtype=float)
        lat = np.asarray(lat, dtype=float)
        t_s = np.asarray(t_s, dtype=float)
        lon, lat, t_s = np.broadcast_arrays(lon, lat, t_s)
        return (
            self._interp(self.U, lon, lat, t_s),
            self._interp(self.V, lon, lat, t_s),
            self._interp(self.U10, lon, lat, t_s),
            self._interp(self.V10, lon, lat, t_s),
        )

    def eff_vel(self, lon, lat, t_s):
        u, v, u10, v10 = self.current_wind(lon, lat, t_s)
        w = config.WIND_DRIFT_FACTOR
        return u + w * u10, v + w * v10


def get_sampler(prefer_netcdf: bool = True):
    """Return the best available field sampler."""
    if prefer_netcdf and config.METEO_NC.exists():
        try:
            return NetCDFSampler(config.METEO_NC)
        except Exception:
            pass
    u0, v0, _ = load_calibration()
    return AnalyticSampler((u0, v0))


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

_CALIBRATION_JSON = config.DATA_DIR / "calibration.json"


def load_calibration() -> tuple[float, float, float]:
    """Return (u0, v0, residual_km) of the calibrated uniform flow."""
    if _CALIBRATION_JSON.exists():
        try:
            d = json.loads(_CALIBRATION_JSON.read_text())
            return float(d["u0"]), float(d["v0"]), float(d["residual_km"])
        except Exception:
            pass
    return (*config.CURRENT_UNIFORM_GUESS, float("nan"))


def calibrate_uniform(verbose: bool = True) -> tuple[float, float, float]:
    """
    Tune the uniform current component so that back-drifting the default slick
    centroid for HINDCAST_HOURS ends exactly at the anchor point.
    """
    u0, v0 = config.CURRENT_UNIFORM_GUESS
    t0 = config.to_epoch_seconds(config.DETECTION_TIME_DEFAULT)
    dur = -config.HINDCAST_HOURS * 3600.0
    residual_km = float("inf")

    for it in range(12):
        sampler = AnalyticSampler((u0, v0))
        traj, _ = integrate_rk4(
            sampler.eff_vel,
            np.array([config.SLICK_LON]),
            np.array([config.SLICK_LAT]),
            t0, dur, config.DRIFT_TIME_STEP_S,
        )
        elon, elat = traj[-1, 0]
        dlat_km = (config.ANCHOR_LAT - elat) * config.KM_PER_DEG_LAT
        dlon_km = (config.ANCHOR_LON - elon) * config.KM_PER_DEG_LON
        residual_km = math.hypot(dlat_km, dlon_km)
        if verbose:
            print(f"  calibration iter {it}: uniform=({u0:+.4f},{v0:+.4f}) m/s "
                  f"-> origin=({elat:.5f},{elon:.5f}), residual={residual_km:.4f} km")
        if residual_km < 1e-3:
            break
        # Negative sign ensures convergence during backward hindcast
        u0 -= dlon_km * 1000.0 / abs(dur)
        v0 -= dlat_km * 1000.0 / abs(dur)

    return u0, v0, residual_km


# ---------------------------------------------------------------------------
# NetCDF writer
# ---------------------------------------------------------------------------

def write_netcdf(path: Path | None = None, verbose: bool = True) -> dict:
    """Generate the 48 h synthetic metocean NetCDF (calibrated)."""
    path = Path(path or config.METEO_NC)
    path.parent.mkdir(parents=True, exist_ok=True)

    u0, v0, residual_km = calibrate_uniform(verbose=verbose)

    lons = np.round(np.arange(config.LON_MIN, config.LON_MAX + 1e-9, config.GRID_STEP_DEG), 6)
    lats = np.round(np.arange(config.LAT_MIN, config.LAT_MAX + 1e-9, config.GRID_STEP_DEG), 6)
    t_start = config.DETECTION_TIME_DEFAULT - timedelta(hours=config.METEO_HISTORY_HOURS)
    nt = int(config.METEO_HISTORY_HOURS * 60 / config.METEO_TIME_STEP_MIN) + 1
    t_s = config.to_epoch_seconds(t_start) + np.arange(nt) * config.METEO_TIME_STEP_MIN * 60.0

    LON3 = lons[None, None, :]
    LAT3 = lats[None, :, None]
    T3 = t_s[:, None, None]

    U, V = current_uv(np.broadcast_to(LON3, (nt, len(lats), len(lons))),
                      np.broadcast_to(LAT3, (nt, len(lats), len(lons))),
                      np.broadcast_to(T3, (nt, len(lats), len(lons))), (u0, v0))
    U10, V10 = wind_uv(np.broadcast_to(LON3, (nt, len(lats), len(lons))),
                       np.broadcast_to(LAT3, (nt, len(lats), len(lons))),
                       np.broadcast_to(T3, (nt, len(lats), len(lons))))

    import netCDF4

    with netCDF4.Dataset(path, "w", format="NETCDF4") as ds:
        ds.title = "AegisX synthetic metocean - Arabian Sea (calibrated)"
        ds.institution = "AegisX synthetic data engine"
        ds.source = "analytic uniform + cold-core eddy + M2 tide + ERA5-like monsoon wind"
        ds.history = "deterministic synthetic generation"
        ds.calibration_residual_km = residual_km
        ds.anchor_lat = config.ANCHOR_LAT
        ds.anchor_lon = config.ANCHOR_LON

        ds.createDimension("time", nt)
        ds.createDimension("lat", len(lats))
        ds.createDimension("lon", len(lons))

        tvar = ds.createVariable("time", "f8", ("time",))
        tvar.standard_name = "time"
        tvar.long_name = "time"
        tvar.units = "seconds since " + config.EPOCH_REF.strftime("%Y-%m-%d %H:%M:%S +00:00")
        tvar.calendar = "standard"

        latvar = ds.createVariable("lat", "f4", ("lat",))
        latvar.standard_name = "latitude"
        latvar.units = "degrees_north"

        lonvar = ds.createVariable("lon", "f4", ("lon",))
        lonvar.standard_name = "longitude"
        lonvar.units = "degrees_east"

        def field_var(name, std_name, data):
            var = ds.createVariable(name, "f4", ("time", "lat", "lon"),
                                    zlib=True, complevel=4)
            var.standard_name = std_name
            var.units = "m s-1"
            var[:] = np.ascontiguousarray(data, dtype=np.float32)
            return var

        field_var("u_current", "eastward_sea_water_velocity", U)
        field_var("v_current", "northward_sea_water_velocity", V)
        wu = field_var("u10", "eastward_wind", U10); wu.setncattr("height", "10 m")
        wv = field_var("v10", "northward_wind", V10); wv.setncattr("height", "10 m")

        tvar[:] = t_s
        latvar[:] = lats
        lonvar[:] = lons

    config.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _CALIBRATION_JSON.write_text(json.dumps(
        {"u0": u0, "v0": v0, "residual_km": residual_km}, indent=2))

    if verbose:
        print(f"  metocean written: {path}  ({nt} times x {len(lats)} lat x {len(lons)} lon)")
        print(f"  calibrated uniform current: u={u0:+.4f} m/s, v={v0:+.4f} m/s "
              f"(anchor residual {residual_km*1000:.1f} m)")

    return {"u0": u0, "v0": v0, "residual_km": residual_km, "path": str(path)}


@dataclass
class FieldCheck:
    origin_lat: float
    origin_lon: float
    residual_km: float


def verify_against_written_file(verbose: bool = True) -> FieldCheck:
    """Independent check: backtrack from the slick using the *written file*."""
    sampler = NetCDFSampler(config.METEO_NC)
    t0 = config.to_epoch_seconds(config.DETECTION_TIME_DEFAULT)
    traj, _ = integrate_rk4(
        sampler.eff_vel,
        np.array([config.SLICK_LON]),
        np.array([config.SLICK_LAT]),
        t0, -config.HINDCAST_HOURS * 3600.0, config.DRIFT_TIME_STEP_S,
    )
    elon, elat = traj[-1, 0]
    res = math.hypot((config.ANCHOR_LAT - elat) * config.KM_PER_DEG_LAT,
                     (config.ANCHOR_LON - elon) * config.KM_PER_DEG_LON)
    if verbose:
        print(f"  netcdf-verification origin: ({elat:.5f}, {elon:.5f}), "
              f"anchor error {res*1000:.0f} m")
    return FieldCheck(elat, elon, res)


if __name__ == "__main__":
    print("AegisX metocean generator")
    write_netcdf()
    verify_against_written_file()
