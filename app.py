import streamlit as st
import pandas as pd
import plotly.express as px
import os
import re
import gdown
from pathlib import Path

st.set_page_config(
    page_title="Marshalls League Data Engine",
    page_icon="⚾",
    layout="wide"
)

FOLDER_ID = "1aJlhryPy5pPqEcbt-EvkEIiLrGwuQtnd"
DOWNLOAD_DIR = Path("./downloaded_games")

@st.cache_data(ttl=600, show_spinner=False)
def load_data_engine(folder_id):
    """Downloads public Drive folder games, parses Sabermetric and Kinematic metrics."""
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download public Drive folder recursively
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    try:
        gdown.download_folder(url=folder_url, output=str(DOWNLOAD_DIR), quiet=True, remaining_ok=True)
    except Exception as e:
        st.error(f"Error connecting to Drive data stream: {e}")

    csv_files = list(DOWNLOAD_DIR.rglob("*.csv"))
    if not csv_files:
        return pd.DataFrame()

    frames = []
    for f in csv_files:
        try:
            df = pd.read_csv(f, low_memory=False)
            if df.empty:
                continue

            # Year Extraction (multi-season intelligence)
            if 'Date' in df.columns and df['Date'].dropna().size > 0:
                df['ParsedDate'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Season_Year'] = df['ParsedDate'].dt.year.fillna(2026).astype(int)
            else:
                match = re.search(r"202\d", f.name)
                df['Season_Year'] = int(match.group(0)) if match else 2026

            df['Game_Source'] = f.stem
            
            # Clean non-pitch sensor artifacts
            if 'RelSpeed' in df.columns:
                df = df[(df['RelSpeed'] >= 35.0) & (df['RelSpeed'] <= 105.0)]
            if 'isOutlier' in df.columns:
                df = df[df['isOutlier'] != True]

            # Sabermetric derived features
            if 'InducedVertBreak' in df.columns and 'VertApprAngle' in df.columns:
                # Ride Efficiency Index (IVB relative to Velocity)
                df['Ride_Per_MPH'] = (df['InducedVertBreak'] / df['RelSpeed']).round(2)

            frames.append(df)
        except Exception:
            continue

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

# ----------------- PRO BRANDING & HEADER -----------------
st.title("⚡ Marshalls League Data Engine")
st.markdown("##### **Created by Jordan Jones** | *Next-Gen Ball Flight Kinematics & Player Development System*")

with st.spinner("Synchronizing game telemetry with Marshalls Cloud Lake..."):
    data = load_data_engine(FOLDER_ID)

if data.empty:
    st.error("No telemetry data could be retrieved. Verify that the Google Drive folder link sharing is set to 'Anyone with the link can view'.")
    st.stop()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.image("https://img.icons8.com/color/96/baseball--v1.png", width=64)
st.sidebar.title("Data Control Room")

# Multi-Season Filter
available_years = sorted(data['Season_Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Season Filter", options=available_years, default=available_years)
df_filtered = data[data['Season_Year'].isin(selected_years)]

# Pitcher Selection
pitchers = sorted([p for p in df_filtered['Pitcher'].dropna().unique() if str(p).strip()])
selected_pitcher = st.sidebar.selectbox("Pitcher Profile", options=["All Arms"] + pitchers)
if selected_pitcher != "All Arms":
    df_filtered = df_filtered[df_filtered['Pitcher'] == selected_pitcher]

# Arsenal Selection
pitch_types = sorted([pt for pt in df_filtered['TaggedPitchType'].dropna().unique() if str(pt).strip()])
selected_pitches = st.sidebar.multiselect("Pitch Arsenal", options=pitch_types, default=pitch_types)
df_filtered = df_filtered[df_filtered['TaggedPitchType'].isin(selected_pitches)]

# ----------------- EXECUTIVE KPI CARDS -----------------
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
kpi1.metric("Tracked Pitches", f"{len(df_filtered):,}")

fb_data = df_filtered[df_filtered['TaggedPitchType'] == 'Fastball']
avg_velo = fb_data['RelSpeed'].mean() if not fb_data.empty else df_filtered['RelSpeed'].mean()
kpi2.metric("Peak FB Velo", f"{fb_data['RelSpeed'].max():.1f} mph" if not fb_data.empty else "N/A")
kpi3.metric("Avg FB Velo", f"{avg_velo:.1f} mph" if pd.notna(avg_velo) else "N/A")

avg_spin = fb_data['SpinRate'].mean() if not fb_data.empty else df_filtered['SpinRate'].mean()
kpi4.metric("Avg Spin Rate", f"{avg_spin:.0f} RPM" if pd.notna(avg_spin) else "N/A")

avg_ext = df_filtered['Extension'].mean() if 'Extension' in df_filtered.columns else None
kpi5.metric("Avg Extension", f"{avg_ext:.1f} ft" if pd.notna(avg_ext) else "N/A")

st.divider()

# ----------------- ANALYTICAL MODULES -----------------
tab_flight, tab_zone, tab_kinematic, tab_yoy = st.tabs([
    "🎯 Ball Flight & Pitch Shapes",
    "📐 Plate Distribution & Zone Matrix",
    "⚡ Kinetic Chain & Energy Transfer",
    "📈 Multi-Season Development"
])

# MODULE 1: FLIGHT & SHAPES
with tab_flight:
    c_plot, c_table = st.columns([2, 1])
    
    with c_plot:
        fig_shape = px.scatter(
            df_filtered,
            x="HorzBreak",
            y="InducedVertBreak",
            color="TaggedPitchType",
            hover_data=["RelSpeed", "SpinRate", "VertApprAngle"],
            title="Pitch Movement Profile: Induced Vertical Break vs. Horizontal Break",
            labels={"HorzBreak": "Horizontal Break (Inches)", "InducedVertBreak": "Induced Vertical Break (Inches)"},
            template="plotly_dark"
        )
        fig_shape.add_hline(y=0, line_dash="dash", line_color="#888888")
        fig_shape.add_vline(x=0, line_dash="dash", line_color="#888888")
        st.plotly_chart(fig_shape, use_container_width=True)
        
    with c_table:
        st.markdown("#### **Arsenal Metrics Breakdown**")
        metric_cols = ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'Extension']
        valid_cols = [col for col in metric_cols if col in df_filtered.columns]
        summary_grid = df_filtered.groupby('TaggedPitchType')[valid_cols].mean().reset_index()
        st.dataframe(summary_grid.round(1), use_container_width=True, hide_index=True)

# MODULE 2: ZONE MATRIX
with tab_zone:
    if 'PlateLocSide' in df_filtered.columns and 'PlateLocHeight' in df_filtered.columns:
        fig_zone = px.scatter(
            df_filtered,
            x="PlateLocSide",
            y="PlateLocHeight",
            color="TaggedPitchType",
            hover_data=["RelSpeed", "VertApprAngle"],
            title="Strike Zone Heat & Location Matrix (Catcher View)",
            labels={"PlateLocSide": "Horizontal Location (ft)", "PlateLocHeight": "Vertical Location (ft)"},
            template="plotly_dark"
        )
        # MLB Standard Zone Outline: Width [-0.83, 0.83], Height [1.5, 3.5]
        fig_zone.add_shape(type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
                           line=dict(color="#00FFCC", width=3))
        fig_zone.update_xaxes(range=[-2.5, 2.5])
        fig_zone.update_yaxes(range=[0, 5])
        st.plotly_chart(fig_zone, use_container_width=True)
    else:
        st.info("Plate location coordinates are unavailable in the selected slice.")

# MODULE 3: KINETICS
with tab_kinematic:
    st.markdown("#### **Biomechanical Rotational Acceleration Engine**")
    if 'PelvisMaxAngularVelocity' in df_filtered.columns and 'ShoulderMaxAngularVelocity' in df_filtered.columns:
        valid_bio = df_filtered[df_filtered['PelvisMaxAngularVelocity'].notna()]
        if not valid_bio.empty:
            fig_kin = px.scatter(
                valid_bio,
                x="PelvisMaxAngularVelocity",
                y="ShoulderMaxAngularVelocity",
                color="TaggedPitchType",
                size="RelSpeed",
                hover_data=["RelSpeed"],
                title="Torso vs. Pelvis Peak Angular Velocity (Rotational Transfer Efficiency)",
                labels={
                    "PelvisMaxAngularVelocity": "Pelvic Peak Angular Velocity (deg/s)",
                    "ShoulderMaxAngularVelocity": "Upper Torso Peak Angular Velocity (deg/s)"
                },
                template="plotly_dark"
            )
            st.plotly_chart(fig_kin, use_container_width=True)
        else:
            st.info("No biomechanical sensor events recorded for this selection.")
    else:
        st.info("Kinetic metrics not found in this dataset.")

# MODULE 4: MULTI-SEASON DEVELOPMENT
with tab_yoy:
    st.markdown("#### **Multi-Year Progression (2026 vs. Future Campaigns)**")
    if len(available_years) > 1:
        fig_yoy = px.box(
            df_filtered,
            x="TaggedPitchType",
            y="RelSpeed",
            color="Season_Year",
            title="Velocity Migration Across Seasons",
            template="plotly_dark"
        )
        st.plotly_chart(fig_yoy, use_container_width=True)
    else:
        st.success("Currently displaying 2026 Inaugural Season data. When 2027 files are placed into the Drive folder next June, side-by-side progression charts will automatically populate here.")
