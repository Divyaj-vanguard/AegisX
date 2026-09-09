"""
AegisX demo orchestrator.

Runs the full attribution pipeline end-to-end and renders the interactive
folium demo map:

    1. ensure the synthetic metocean NetCDF exists (calibrated ensemble)
    2. run the Lagrangian hindcast (OpenDrift, RK4 fallback) from the slick
    3. ensure the synthetic AIS database (target vessel + decoys)
    4. spatially/temporally match ships against the hindcast origin and
       score behavioural anomalies
    5. render output/demo_map.html

Usage:
    python main.py                       # full demo with default slick
    python main.py --engine rk4          # force the built-in RK4 engine
    python main.py --slick-lat 18.9 --slick-lon 72.5
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

import config
import metfield
import physics_engine
import spatial_matcher

OUTPUT_DIR = config.BASE_DIR / "output"
DEFAULT_MAP = OUTPUT_DIR / "demo_map.html"


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class PipelineResult:
    def __init__(self, hindcast, reports, slick_lat, slick_lon, detection):
        self.hindcast = hindcast
        self.reports = reports
        self.slick_lat = slick_lat
        self.slick_lon = slick_lon
        self.detection = detection


def run_pipeline(slick_lat: float = config.SLICK_LAT,
                 slick_lon: float = config.SLICK_LON,
                 detection: datetime | None = None,
                 engine: str | None = None,
                 verbose: bool = True) -> PipelineResult:
    detection = detection or config.DETECTION_TIME_DEFAULT

    if not config.METEO_NC.exists():
        if verbose:
            print("[1/4] metocean NetCDF missing - generating calibrated field")
        metfield.write_netcdf(verbose=verbose)
    elif verbose:
        print(f"[1/4] metocean: {config.METEO_NC.name} (cached)")

    if verbose:
        print(f"[2/4] hindcast: slick=({slick_lat}, {slick_lon}) "
              f"detected {detection:%Y-%m-%d %H:%MZ}")
    hindcast = physics_engine.run_hindcast(slick_lat, slick_lon, detection,
                                           engine=engine)
    if verbose:
        print(f"      engine={hindcast.engine}  "
              f"origin=({hindcast.origin_lat:.5f}, {hindcast.origin_lon:.5f})  "
              f"t_origin={hindcast.t_origin:%Y-%m-%d %H:%MZ}  "
              f"ellipse {hindcast.ellipse['semi_major_km']}x"
              f"{hindcast.ellipse['semi_minor_km']} km")

    db = spatial_matcher.ensure_ais_db()
    if verbose:
        print(f"[3/4] AIS picture: {db.name}")

    tracks, vessels = spatial_matcher.load_ais()
    reports = spatial_matcher.match_candidates(hindcast, tracks, vessels)
    if verbose:
        print("[4/4] attribution ranking:")
        for r in reports:
            print(f"      {r.score:3d}  {r.verdict:<16s} {r.name:<22s} "
                  f"MMSI {r.mmsi}  closest {r.closest_km:5.2f} km  "
                  f"gap {r.max_gap_h:.1f} h  min SOG {r.min_sog_kn:.1f} kn")

    return PipelineResult(hindcast, reports, slick_lat, slick_lon, detection)


# ---------------------------------------------------------------------------
# Folium map
# ---------------------------------------------------------------------------

_SCORE_COLORS = [(70, "#d62728"), (40, "#ff7f0e"), (25, "#1f77b4")]
_CLEAR_COLOR = "#7f7f7f"


def _score_color(score: int) -> str:
    for thresh, color in _SCORE_COLORS:
        if score >= thresh:
            return color
    return _CLEAR_COLOR


def _vessel_popup(r: spatial_matcher.VesselReport) -> str:
    rows = [
        f"<b>{r.name}</b>  ({r.ship_type})",
        f"MMSI {r.mmsi} &middot; flag {r.flag}",
        f"<b>score {r.score}/100 &mdash; {r.verdict}</b>",
        f"closest approach: {r.closest_km:.2f} km",
        f"max AIS gap: {r.max_gap_h:.1f} h &middot; "
        f"SOG min/cruise: {r.min_sog_kn:.1f}/{r.cruise_sog_kn:.1f} kn",
    ]
    if r.flags:
        rows.append("anomalies: " + "; ".join(r.flags))
    return "<br>".join(rows)


def build_map(res: PipelineResult, out_path: Path | str = DEFAULT_MAP) -> Path:
    import folium
    from folium import plugins

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    h = res.hindcast
    center_lat = 0.5 * (res.slick_lat + h.origin_lat)
    center_lon = 0.5 * (res.slick_lon + h.origin_lon)

    m = folium.Map(location=[center_lat, center_lon], zoom_start=9,
                   tiles="CartoDB positron")
    folium.TileLayer("OpenStreetMap", name="OpenStreetMap").add_to(m)

    fg_slick = folium.FeatureGroup("Slick detection", show=True)
    fg_drift = folium.FeatureGroup("Hindcast drift", show=True)
    fg_origin = folium.FeatureGroup("Origin estimate", show=True)
    fg_ships = folium.FeatureGroup("AIS traffic", show=True)

    # -- slick ----------------------------------------------------------------
    folium.Marker(
        [res.slick_lat, res.slick_lon],
        icon=folium.Icon(color="red", icon="tint", prefix="fa"),
        popup=(f"Oil slick detected {res.detection:%Y-%m-%d %H:%M}Z<br>"
               f"centroid ({res.slick_lat:.4f}, {res.slick_lon:.4f})"),
        tooltip="Detected slick centroid",
    ).add_to(fg_slick)
    folium.Circle([res.slick_lat, res.slick_lon], radius=config.SEED_RADIUS_M,
                  color="red", weight=1, fill=True, fill_opacity=0.15).add_to(fg_slick)

    # -- backward drift -------------------------------------------------------
    path = list(zip(h.path_lat.tolist(), h.path_lon.tolist()))
    plugins.AntPath(path, color="#084594", weight=3, delay=1200,
                    tooltip="back-drift trajectory (detection -> origin)").add_to(fg_drift)
    folium.PolyLine(path, color="#084594", weight=1, opacity=0.4).add_to(fg_drift)
    # time-tagged dots every 6 h along the back-drift
    step = max(1, int(6 * 3600 / abs(config.DRIFT_TIME_STEP_S)))
    for i in range(0, len(path), step):
        t = h.path_times[i]
        folium.CircleMarker(path[i], radius=3, color="#084594", fill=True,
                            tooltip=f"t {t:%d %b %H:%MZ}").add_to(fg_drift)
    # particle endpoint cloud
    for lat, lon in zip(h.endpoints_lat.tolist()[::4], h.endpoints_lon.tolist()[::4]):
        folium.CircleMarker([lat, lon], radius=2, color="#2ca02c",
                            fill=True, fill_opacity=0.5, weight=0).add_to(fg_origin)

    # -- origin + 95 % ellipse -------------------------------------------------
    ring_ll = [(lat, lon) for lon, lat in h.ellipse_lonlat]
    folium.Polygon(
        ring_ll, color="#2ca02c", weight=2, dash_array="6",
        fill=True, fill_opacity=0.08,
        tooltip=(f"95% origin ellipse "
                 f"{h.ellipse['semi_major_km']}x{h.ellipse['semi_minor_km']} km"),
    ).add_to(fg_origin)
    folium.Marker(
        [h.origin_lat, h.origin_lon],
        icon=folium.Icon(color="green", icon="flag", prefix="fa"),
        popup=(f"<b>Hindcast origin</b> ({h.origin_lat:.4f}, {h.origin_lon:.4f})<br>"
               f"t_origin {h.t_origin:%Y-%m-%d %H:%M}Z &plusmn; "
               f"{config.ORIGIN_TIME_WINDOW_H} h"),
        tooltip="Hindcast origin",
    ).add_to(fg_origin)
    folium.Marker(
        [config.ANCHOR_LAT, config.ANCHOR_LON],
        icon=folium.Icon(color="purple", icon="star", prefix="fa"),
        popup=(f"Scenario anchor (ground truth) "
               f"({config.ANCHOR_LAT:.4f}, {config.ANCHOR_LON:.4f})"),
        tooltip="Scenario anchor (ground truth)",
    ).add_to(fg_origin)

    # -- AIS traffic ----------------------------------------------------------
    t0 = h.t_origin
    lb_h = config.ANOMALY_LOOKBACK_H
    for r in res.reports:
        color = _score_color(r.score)
        g = r.track
        if len(g) < 2:
            continue
        pts = list(zip(g["lat"].tolist(), g["lon"].tolist()))
        # full +/-12 h track, faint
        folium.PolyLine(pts, color=color, weight=1.2, opacity=0.35,
                        tooltip=f"{r.name} (score {r.score})").add_to(fg_ships)
        # emphasised segment around t_origin
        win = g[(g["ts"] >= t0 - pd.Timedelta(hours=lb_h))
                & (g["ts"] <= t0 + pd.Timedelta(hours=lb_h))]
        if len(win) >= 2:
            pts_w = list(zip(win["lat"].tolist(), win["lon"].tolist()))
            folium.PolyLine(
                pts_w, color=color, weight=3.2 if r.score > 0 else 2.0,
                opacity=0.95 if r.score > 0 else 0.6,
                popup=folium.Popup(_vessel_popup(r), max_width=320),
                tooltip=f"{r.name}  (score {r.score})",
            ).add_to(fg_ships)
            folium.CircleMarker(
                pts_w[0], radius=4, color=color, fill=True,
                tooltip=f"{r.name}<br>{win['ts'].iloc[0]:%d %b %H:%MZ}  "
                        f"SOG {win['sog'].iloc[0]:.1f} kn",
            ).add_to(fg_ships)
            folium.CircleMarker(
                pts_w[-1], radius=4, color=color, fill=True,
                tooltip=f"{r.name}<br>{win['ts'].iloc[-1]:%d %b %H:%MZ}  "
                        f"SOG {win['sog'].iloc[-1]:.1f} kn",
            ).add_to(fg_ships)
        # blackout segment: dashed red chord across the gap
        if r.gap_entry_lat is not None and r.gap_exit_lat is not None:
            folium.PolyLine(
                [(r.gap_entry_lat, r.gap_entry_lon),
                 (r.gap_exit_lat, r.gap_exit_lon)],
                color="#d62728", weight=2.5, dash_array="8 6",
                tooltip=f"AIS blackout {r.max_gap_h:.1f} h",
            ).add_to(fg_ships)

    fg_slick.add_to(m)
    fg_drift.add_to(m)
    fg_origin.add_to(m)
    fg_ships.add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)

    # -- title / legend ---------------------------------------------------------
    title = folium.Element(f"""
    <div style="position: fixed; top: 12px; left: 60px; z-index: 9999;
                background: rgba(255,255,255,0.92); padding: 10px 14px;
                border: 1px solid #888; border-radius: 6px;
                font-family: sans-serif; font-size: 13px; max-width: 340px">
      <b>AegisX &mdash; oil spill attribution demo</b><br>
      slick detected {res.detection:%Y-%m-%d %H:%M}Z &middot;
      engine <i>{h.engine}</i><br>
      t_origin {h.t_origin:%Y-%m-%d %H:%M}Z &plusmn;{config.ORIGIN_TIME_WINDOW_H} h<br>
      <span style="color:#d62728">&#9632;</span> prime suspect (score&ge;70)
      <span style="color:#ff7f0e">&#9632;</span> suspicious (40+)
      <span style="color:#1f77b4">&#9632;</span> witness (25+)
      <span style="color:#7f7f7f">&#9632;</span> clear
    </div>""")
    m.get_root().html.add_child(title)

    m.fit_bounds([
        [min(res.slick_lat, h.origin_lat, config.ANCHOR_LAT) - 0.25,
         min(res.slick_lon, h.origin_lon, config.ANCHOR_LON) - 0.35],
        [max(res.slick_lat, h.origin_lat, config.ANCHOR_LAT) + 0.25,
         max(res.slick_lon, h.origin_lon, config.ANCHOR_LON) + 0.35],
    ])

    m.save(str(out_path))
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="AegisX demo pipeline")
    p.add_argument("--engine", choices=["opendrift", "rk4"], default=None,
                   help="force physics engine (default: auto, OpenDrift first)")
    p.add_argument("--slick-lat", type=float, default=config.SLICK_LAT)
    p.add_argument("--slick-lon", type=float, default=config.SLICK_LON)
    p.add_argument("--detection", default=None,
                   help="detection time ISO-8601 (default: config value)")
    p.add_argument("--output", default=str(DEFAULT_MAP),
                   help=f"map output path (default: {DEFAULT_MAP})")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    detection = None
    if args.detection:
        detection = datetime.fromisoformat(args.detection.replace("Z", "+00:00"))
        if detection.tzinfo is None:
            detection = detection.replace(tzinfo=timezone.utc)

    res = run_pipeline(args.slick_lat, args.slick_lon, detection,
                       engine=args.engine)

    out = build_map(res, args.output)

    # demo self-check: the planted target must be the top suspect
    suspects = spatial_matcher.top_suspects(res.reports)
    top = suspects[0] if suspects else None
    ok = (top is not None and top.mmsi == config.TARGET_MMSI
          and top.score >= config.SCORE_AIS_BLACKOUT
          + config.SCORE_SPEED_DROP + config.SCORE_PROXIMITY)
    print()
    if ok:
        print(f"ATTRIBUTION OK: {top.name} (MMSI {top.mmsi}) "
              f"score {top.score}/100 -> {top.verdict}")
        for f in top.flags:
            print(f"   - {f}")
    else:
        name = top.name if top else "none"
        print(f"WARNING: expected {config.TARGET_NAME} as top suspect, got {name}")
    print(f"\nDemo map written: {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
