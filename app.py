import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import io
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(
    page_title="Marshalls League Data Engine",
    page_icon="⚾",
    layout="wide"
)

MANIFEST_SHEET_ID = "1Xc3lx4ybIfp9R14ROhCWOD1RpnKIhbNYU76dYQUZdow"

def fetch_single_csv(args):
    file_id, game_name, season_year = args
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        res = requests.get(url, timeout=12)
        if res.status_code == 200 and b"<html" not in res.content[:80].lower():
            df = pd.read_csv(io.BytesIO(res.content), low_memory=False)
            if not df.empty and 'TaggedPitchType' in df.columns:
                df['Season_Year'] = int(season_year)
                df['Game_Source'] = str(game_name)
                
                # Filter launch monitor tracking artifacts
                if 'RelSpeed' in df.columns:
                    df = df[(df['RelSpeed'] >= 35.0) & (df['RelSpeed'] <= 106.0)]
                if 'isOutlier' in df.columns:
                    df = df[df['isOutlier'] != True]
                return df
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def load_marshalls_telemetry(sheet_id):
    sheet_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv"
    try:
        manifest_df = pd.read_csv(sheet_url)
    except Exception as e:
        st.error(f"Could not read Game Manifest sheet: {e}")
        return pd.DataFrame()

    tasks = [
        (str(row['File_ID']).strip(), str(row['Game_Name']).strip(), row['Season'])
        for _, row in manifest_df.iterrows()
        if pd.notna(row['File_ID'])
    ]

    frames = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(fetch_single_csv, tasks)
        for res in results:
            if res is not None:
                frames.append(res)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        # Normalize break metrics to realistic baseball boundaries
        if 'InducedVertBreak' in combined.columns:
            combined = combined[(combined['InducedVertBreak'] >= -35) & (combined['InducedVertBreak'] <= 35)]
        if 'HorzBreak' in combined.columns:
            combined = combined[(combined['HorzBreak'] >= -35) & (combined['HorzBreak'] <= 35)]
        return combined
    return pd.DataFrame()

# ----------------- PRO BRANDING & HEADER -----------------
st.title("⚡ Marshalls League Data Engine")
st.markdown("##### **Created by Jordan Jones** | *Next-Gen Ball Flight Kinematics & Player Development System*")

with st.spinner("Streaming full multi-game telemetry lake..."):
    data = load_marshalls_telemetry(MANIFEST_SHEET_ID)

if data.empty:
    st.warning("Telemetry is synchronizing. Please confirm sheet access.")
    st.stop()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🎯 Data Control Room")

# Mode Switch: Pitcher vs Hitter
view_mode = st.sidebar.radio("Analysis Mode", ["⚾ Pitcher Diagnostics", "💥 Hitter Performance & Exit Velo"])

# Season Filter
available_years = sorted(data['Season_Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Season Filter", options=available_years, default=available_years)
df = data[data['Season_Year'].isin(selected_years)]

# =====================================================================
# PITCHER DIAGNOSTICS VIEW
# =====================================================================
if view_mode == "⚾ Pitcher Diagnostics":
    if 'Pitcher' in df.columns:
        pitchers = sorted([p for p in df['Pitcher'].dropna().unique() if str(p).strip()])
        selected_pitcher = st.sidebar.selectbox("Pitcher Profile", options=["All Pitchers"] + pitchers)
        if selected_pitcher != "All Pitchers":
            df = df[df['Pitcher'] == selected_pitcher]

    pitch_types = sorted([pt for pt in df['TaggedPitchType'].dropna().unique() if str(pt).strip()])
    selected_pitches = st.sidebar.multiselect("Pitch Arsenal", options=pitch_types, default=pitch_types)
    df = df[df['TaggedPitchType'].isin(selected_pitches)]

    # Pitcher KPIs
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Tracked Pitches", f"{len(df):,}")
    fb_df = df[df['TaggedPitchType'] == 'Fastball']
    avg_velo = fb_df['RelSpeed'].mean() if not fb_df.empty else df['RelSpeed'].mean()
    k2.metric("Peak FB Velo", f"{fb_df['RelSpeed'].max():.1f} mph" if not fb_df.empty else "N/A")
    k3.metric("Avg FB Velo", f"{avg_velo:.1f} mph" if pd.notna(avg_velo) else "N/A")
    avg_spin = fb_df['SpinRate'].mean() if not fb_df.empty else df['SpinRate'].mean()
    k4.metric("Avg Spin Rate", f"{avg_spin:.0f} RPM" if pd.notna(avg_spin) else "N/A")
    avg_ext = df['Extension'].mean() if 'Extension' in df.columns else None
    k5.metric("Avg Extension", f"{avg_ext:.1f} ft" if pd.notna(avg_ext) else "N/A")

    st.divider()

    tab_flight, tab_zone, tab_kinematic = st.tabs([
        "🎯 Ball Flight & Pitch Shapes",
        "📐 Plate Distribution & Zone Matrix",
        "⚡ Kinetic Chain & Energy Transfer"
    ])

    with tab_flight:
        c_plot, c_table = st.columns([2, 1])
        with c_plot:
            fig_shape = px.scatter(
                df,
                x="HorzBreak",
                y="InducedVertBreak",
                color="TaggedPitchType",
                hover_data=["RelSpeed", "SpinRate", "VertApprAngle"],
                title="Pitch Movement Profile: IVB vs. HB (Inches)",
                labels={"HorzBreak": "Horizontal Break (Inches)", "InducedVertBreak": "Induced Vertical Break (Inches)"},
                template="plotly_dark"
            )
            fig_shape.update_xaxes(range=[-25, 25])
            fig_shape.update_yaxes(range=[-25, 25])
            fig_shape.add_hline(y=0, line_dash="dash", line_color="#888888")
            fig_shape.add_vline(x=0, line_dash="dash", line_color="#888888")
            st.plotly_chart(fig_shape, use_container_width=True)
            
        with c_table:
            st.markdown("#### **Arsenal Metrics Breakdown**")
            metric_cols = ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'Extension']
            valid_cols = [c for c in metric_cols if c in df.columns]
            summary_grid = df.groupby('TaggedPitchType')[valid_cols].mean().reset_index()
            st.dataframe(summary_grid.round(1), use_container_width=True, hide_index=True)

    with tab_zone:
        if 'PlateLocSide' in df.columns and 'PlateLocHeight' in df.columns:
            fig_zone = px.scatter(
                df,
                x="PlateLocSide",
                y="PlateLocHeight",
                color="TaggedPitchType",
                hover_data=["RelSpeed"],
                title="Strike Zone Location (Catcher Perspective)",
                labels={"PlateLocSide": "Horizontal Location (ft)", "PlateLocHeight": "Vertical Location (ft)"},
                template="plotly_dark"
            )
            fig_zone.add_shape(type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
                               line=dict(color="#00FFCC", width=3))
            fig_zone.update_xaxes(range=[-2.5, 2.5])
            fig_zone.update_yaxes(range=[0, 5])
            st.plotly_chart(fig_zone, use_container_width=True)

    with tab_kinematic:
        if 'PelvisMaxAngularVelocity' in df.columns and 'ShoulderMaxAngularVelocity' in df.columns:
            valid_bio = df[df['PelvisMaxAngularVelocity'].notna()]
            if not valid_bio.empty:
                fig_kin = px.scatter(
                    valid_bio,
                    x="PelvisMaxAngularVelocity",
                    y="ShoulderMaxAngularVelocity",
                    color="TaggedPitchType",
                    size="RelSpeed",
                    title="Rotational Sequencing: Pelvis vs. Shoulder Peak Angular Velocity",
                    labels={"PelvisMaxAngularVelocity": "Pelvis Max Angular Vel (deg/s)", "ShoulderMaxAngularVelocity": "Torso Max Angular Vel (deg/s)"},
                    template="plotly_dark"
                )
                st.plotly_chart(fig_kin, use_container_width=True)

# =====================================================================
# HITTER PERFORMANCE & EXIT VELO VIEW
# =====================================================================
else:
    # Filter by Batter
    if 'Batter' in df.columns:
        batters = sorted([b for b in df['Batter'].dropna().unique() if str(b).strip()])
        selected_batter = st.sidebar.selectbox("Hitter Profile", options=["All Hitters"] + batters)
        if selected_batter != "All Hitters":
            df = df[df['Batter'] == selected_batter]

    # Filter by Batter Side
    if 'BatterSide' in df.columns:
        sides = [s for s in df['BatterSide'].dropna().unique() if str(s).strip()]
        selected_side = st.sidebar.multiselect("Batter Handedness", options=sides, default=sides)
        df = df[df['BatterSide'].isin(selected_side)]

    # Batted ball subset
    batted_df = df[df['ExitSpeed'].notna() & (df['ExitSpeed'] > 40)]
    hard_hit = batted_df[batted_df['ExitSpeed'] >= 95.0]
    sweet_spot = batted_df[(batted_df['Angle'] >= 8.0) & (batted_df['Angle'] <= 32.0)]

    # Hitter Executive KPIs
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Batted Ball Events", f"{len(batted_df):,}")
    max_ev = batted_df['ExitSpeed'].max() if not batted_df.empty else None
    h2.metric("Max Exit Velocity", f"{max_ev:.1f} mph" if pd.notna(max_ev) else "N/A")
    avg_ev = batted_df['ExitSpeed'].mean() if not batted_df.empty else None
    h3.metric("Avg Exit Velocity", f"{avg_ev:.1f} mph" if pd.notna(avg_ev) else "N/A")
    hh_pct = (len(hard_hit) / len(batted_df) * 100) if len(batted_df) > 0 else 0
    h4.metric("Hard-Hit % (≥95)", f"{hh_pct:.1f}%")
    sw_pct = (len(sweet_spot) / len(batted_df) * 100) if len(batted_df) > 0 else 0
    h5.metric("Sweet-Spot % (8°-32°)", f"{sw_pct:.1f}%")

    st.divider()

    h_tab1, h_tab2, h_tab3 = st.tabs([
        "🚀 Exit Velo & Launch Angle",
        "🗺️ Field Spray Chart",
        "⚡ Swing Biomechanics & Rotational Timing"
    ])

    with h_tab1:
        c_la, c_ev_dist = st.columns([2, 1])
        with c_la:
            if not batted_df.empty and 'Angle' in batted_df.columns:
                fig_la = px.scatter(
                    batted_df,
                    x="Angle",
                    y="ExitSpeed",
                    color="TaggedPitchType" if 'TaggedPitchType' in batted_df.columns else None,
                    hover_data=["Distance", "PlayResult"] if "PlayResult" in batted_df.columns else ["Distance"],
                    title="Exit Velocity vs. Launch Angle Profile",
                    labels={"Angle": "Launch Angle (Degrees)", "ExitSpeed": "Exit Velocity (MPH)"},
                    template="plotly_dark"
                )
                # Shaded Sweet Spot / Barrel Zone (8° to 32°)
                fig_la.add_vrect(x0=8, x1=32, fillcolor="#00FFCC", opacity=0.1, line_width=0, annotation_text="Sweet Spot (8°-32°)")
                fig_la.add_hline(y=95, line_dash="dash", line_color="#FF4B4B", annotation_text="95+ MPH Hard-Hit")
                st.plotly_chart(fig_la, use_container_width=True)
            else:
                st.info("No recorded batted ball tracking coordinates for this hitter selection.")
                
        with c_ev_dist:
            st.markdown("#### **Batted Ball Outcomes**")
            if 'HitType' in batted_df.columns:
                hit_types = batted_df['HitType'].value_counts().reset_index()
                hit_types.columns = ['Batted Ball Type', 'Count']
                st.dataframe(hit_types, use_container_width=True, hide_index=True)
            if 'PlayResult' in batted_df.columns:
                st.markdown("#### **Play Results**")
                results = batted_df['PlayResult'].value_counts().reset_index()
                results.columns = ['Result', 'Count']
                st.dataframe(results, use_container_width=True, hide_index=True)

    with h_tab2:
        if not batted_df.empty and 'Direction' in batted_df.columns and 'Distance' in batted_df.columns:
            # Polar Spray Chart
            fig_spray = px.scatter_polar(
                batted_df,
                r="Distance",
                theta="Direction",
                color="ExitSpeed",
                size="ExitSpeed",
                range_r=[0, 450],
                start_angle=0,
                direction="counterclockwise",
                title="Field Spray & Contact Distance Map",
                template="plotly_dark"
            )
            st.plotly_chart(fig_spray, use_container_width=True)
        else:
            st.info("Spray chart direction coordinates are unavailable.")

    with h_tab3:
        st.markdown("#### **WIN Reality Swing Kinematics & Kinetic Timing**")
        bio_fields = [
            'Max Hip-Shoulder Separation',
            'Hip-Shoulder Separation at Contact',
            'Pelvis Load',
            'Stride Length',
            'ForwardBendAtContact'
        ]
        available_bio = [c for c in bio_fields if c in df.columns]
        if available_bio and df[available_bio[0]].notna().any():
            avg_swing_bio = df[available_bio].mean().reset_index()
            avg_swing_bio.columns = ['Kinematic Segment Metric', 'Average Value']
            st.dataframe(avg_swing_bio.round(2), use_container_width=True, hide_index=True)
            
            if 'Max Hip-Shoulder Separation' in df.columns and 'Hip-Shoulder Separation at Contact' in df.columns:
                fig_sep = px.scatter(
                    df[df['Max Hip-Shoulder Separation'].notna()],
                    x="Max Hip-Shoulder Separation",
                    y="Hip-Shoulder Separation at Contact",
                    color="ExitSpeed" if 'ExitSpeed' in df.columns else None,
                    title="Hip-Shoulder Separation: Peak Load vs. Contact",
                    template="plotly_dark"
                )
                st.plotly_chart(fig_sep, use_container_width=True)
        else:
            st.info("Swing kinematic tracking metrics not found for the selected swings.")
