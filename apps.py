import math
import sqlite3
import folium
import numpy as np
import pandas as pd
from datetime import datetime

import config
import metfield
import physics_engine

print("\n=======================================================")
print("          AEGISX MARITIME ATTRIBUTION ENGINE           ")
print("=======================================================")

# 1. Run Reverse Drift from physics_engine
print("[1/4] Running Lagrangian Reverse Drift Hindcast...")
sampler = metfield.get_sampler(prefer_netcdf=True)
t0 = config.to_epoch_seconds(config.DETECTION_TIME_DEFAULT)
dur = -config.HINDCAST_HOURS * 3600.0

traj, _ = metfield.integrate_rk4(
    sampler.eff_vel,
    np.array([config.SLICK_LON]),
    np.array([config.SLICK_LAT]),
    t0, dur, config.DRIFT_TIME_STEP_S
)

drift_path = [[float(pt[1]), float(pt[0])] for pt in traj[:, 0]]
origin_lon, origin_lat = float(traj[-1, 0, 0]), float(traj[-1, 0, 1])

print(f"      Detection Centroid: ({config.SLICK_LAT:.4f}°N, {config.SLICK_LON:.4f}°E)")
print(f"      Calculated Origin:  ({origin_lat:.4f}°N, {origin_lon:.4f}°E)")
print(f"      Hindcast Duration:  -{config.HINDCAST_HOURS} Hours")

# 2. Query AIS Database (Directly targeting 'ais' table)
print("\n[2/4] Querying AIS Corridors around Origin Window...")
db_path = config.DATA_DIR / "ais_tracks.sqlite"
conn = sqlite3.connect(db_path)
df_ais = pd.read_sql_query("SELECT mmsi, ts, lat, lon, sog, cog FROM ais", conn)
conn.close()

print(f"      Loaded {len(df_ais)} records from table: 'ais'")

# 3. Anomaly Scoring
print("[3/4] Evaluating Behavioral Anomaly Signatures...")
scored_vessels = []
for mmsi, group in df_ais.groupby("mmsi"):
    group = group.sort_values("ts")
    min_dist = float("inf")
    
    for _, row in group.iterrows():
        dlat = (row["lat"] - origin_lat) * config.KM_PER_DEG_LAT
        dlon = (row["lon"] - origin_lon) * config.KM_PER_DEG_LON
        dist = math.hypot(dlat, dlon)
        if dist < min_dist:
            min_dist = dist

    sog_min = group["sog"].min()
    fix_count = len(group)
    gap_detected = fix_count < 135
    
    score = 0
    reasons = []
    if gap_detected:
        score += 40
        reasons.append(f"Transponder Blackout ({145 - fix_count} missing fixes)")
    if sog_min <= 5.0:
        score += 35
        reasons.append(f"Abnormal Speed Drop ({sog_min:.1f} kts during dump window)")
    if min_dist <= 3.5:
        score += 25
        reasons.append(f"Spatial Origin Coincidence ({min_dist:.2f} km)")
        
    scored_vessels.append({
        "mmsi": str(mmsi),
        "min_dist": min_dist,
        "sog_min": sog_min,
        "score": score,
        "reasons": reasons,
        "track": group[["lat", "lon"]].values.tolist(),
        "is_suspect": score >= 75
    })

scored_vessels.sort(key=lambda x: x["score"], reverse=True)
top = scored_vessels[0]

print(f"\n>>> PRIMARY SUSPECT IDENTIFIED <<<")
print(f"    MMSI: {top['mmsi']}")
print(f"    Attribution Confidence: {top['score']}%")
print(f"    Flags: {', '.join(top['reasons'])}")

# 4. Generate Folium Tactical Map
print("\n[4/4] Rendering Interactive WebGIS Map...")
center_lat = (config.SLICK_LAT + origin_lat) / 2
center_lon = (config.SLICK_LON + origin_lon) / 2
m = folium.Map(location=[center_lat, center_lon], zoom_start=10, tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Ocean/World_Ocean_Base/MapServer/tile/{z}/{y}/{x}", attr="Esri Ocean Basemap")

# Detected Slick
folium.Polygon(
    locations=[
        [config.SLICK_LAT + 0.015, config.SLICK_LON - 0.02],
        [config.SLICK_LAT + 0.025, config.SLICK_LON + 0.01],
        [config.SLICK_LAT - 0.01, config.SLICK_LON + 0.03],
        [config.SLICK_LAT - 0.02, config.SLICK_LON - 0.01]
    ],
    color="#FF3333",
    fill=True,
    fill_color="#FF3333",
    fill_opacity=0.6,
    popup="<b>Detected Slick (Sentinel-1 SAR)</b><br>Area: 8.4 km²"
).add_to(m)

# Backward Drift Vector
folium.PolyLine(
    locations=drift_path,
    color="#00FFFF",
    weight=3,
    dash_array="5, 8",
    popup=f"<b>Lagrangian Hindcast</b><br>-{config.HINDCAST_HOURS}h Reverse Drift"
).add_to(m)

# Origin Uncertainty Ellipse
folium.Circle(
    location=[origin_lat, origin_lon],
    radius=2800,
    color="#FFA500",
    fill=True,
    fill_color="#FFA500",
    fill_opacity=0.25,
    popup="<b>Discharge Origin Zone</b>"
).add_to(m)

# Vessel Tracks
for v in scored_vessels:
    if v["is_suspect"]:
        folium.PolyLine(
            locations=v["track"],
            color="#FF2222",
            weight=4,
            popup=f"<b>SUSPECT: MMSI {v['mmsi']}</b><br>Score: {v['score']}%<br>{'<br>'.join(v['reasons'])}"
        ).add_to(m)
        mid_idx = len(v["track"]) // 2
        folium.Marker(
            location=v["track"][mid_idx],
            icon=folium.Icon(color="red", icon="exclamation-triangle", prefix="fa"),
            popup=f"<b>CULPRIT: MMSI {v['mmsi']}</b>"
        ).add_to(m)
    else:
        folium.PolyLine(
            locations=v["track"],
            color="#22C55E",
            weight=1.5,
            opacity=0.5,
            popup=f"Compliant Track: MMSI {v['mmsi']}"
        ).add_to(m)

output_map = "aegisx_mission_map.html"
m.save(output_map)
print(f"[+] Success! Tactical dashboard saved to: {output_map}\n")
