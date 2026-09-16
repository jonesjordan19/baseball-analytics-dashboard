import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io
import re

st.set_page_config(
    page_title="Marshalls League Data Engine",
    page_icon="⚾",
    layout="wide"
)

# Registry of files in the Google Drive folder
GAME_FILES = [
    {"id": "1oychbUr3bienpfyRq-oa7jerfCfAkI0P", "name": "7-28-2026_04-08PM_Anchors 2026 x Red Hots 2026.csv"},
    {"id": "1QPFkec3o2HKK_64F-2cRwzucpOZByfpT", "name": "7-27-2026_08-05PM_Royals 2026 x Red Hots 2026.csv"},
    {"id": "1tGLmuS3b7nFsrLnGDpCwZK5mMlL2hloo", "name": "7-27-2026_03-43PM_Red Hots 2026 x Blue Crew 2026.csv"},
    {"id": "1RmNZW7fKj87TrBZ1dP6bxn6UNOmz86tr", "name": "7-26-2026_07-24PM_Anchors 2026 x Royals 2026.csv"},
    {"id": "1BYQuWzCFz9TeGccFg7mAdm5F_fZ6qT4V", "name": "7-26-2026_03-42PM_Red Hots 2026 x Blue Crew 2026.csv"},
    {"id": "1_vfdohpD83P-aENXity-RYD7f6Vf7qDV", "name": "7-25-2026_06-35PM_Anchors 2026 x Red Hots 2026.csv"},
    {"id": "1DWAQUJyuelU_RlQTkFS2bYCpx16vYHBl", "name": "7-25-2026_03-48PM_Royals 2026 x Blue Crew 2026.csv"},
    {"id": "1ZEmd4aSsfm-mX9lVrPaSoqBHXqAEsjxF", "name": "7-24-2026_03-51PM_Royals 2026 x Red Hots 2026.csv"},
    {"id": "1Pxj3BfAyDZFKNCrgW7PFKtewzz8akyeo", "name": "7-24-2026_08-27PM_Anchors 2026 x Blue Crew 2026.csv"},
    {"id": "1C-RWWz4B8qwCvfjaX4_fnnIX2ZlvFzu_", "name": "7-22-2026_07-13PM_Royals 2026 x Blue Crew 2026.csv"},
    {"id": "1N-PJsSojyacKQpa3wOzvgFiOk7CDyrlW", "name": "7-22-2026_03-36PM_Red Hots 2026 x Anchors 2026.csv"},
    {"id": "1wTQanptOBiRPPQw7P5tabw--w0XexQ3S", "name": "7-21-2026_03-40PM_Royals 2026 x Red Hots 2026.csv"},
    {"id": "1ynBLWRxsL9zVMTWjGDL0_7XLeHxnf1lt", "name": "7-21-2026_07-17PM_Blue Crew 2026 x Anchors 2026.csv"},
    {"id": "12KwXvDNLYocKH7zN5K5BETqI6uN42sfX", "name": "7-20-2026_07-06PM_Red Hots 2026 x Blue Crew 2026.csv"},
    {"id": "1XXaQj7yHImg3xwJ6WmH8tsAGbF8dXU5q", "name": "7-20-2026_03-54PM_Anchors 2026 x Royals 2026.csv"},
    {"id": "15iMsmm-i7PhNXWCmF-HMxZt9QCl7wMmV", "name": "7-18-2026_06-40PM_Red Hots 2026 x Puerto Rico.csv"},
    {"id": "16Y99weUeiqwM2qUYY36YwxxK2VnTh8k5", "name": "7-18-2026_03-35PM_Blue Crew 2026 x Red Hots 2026.csv"},
    {"id": "1FeHyA34Uhy2cQ1P0SGPp4TRgdH_lQGjB", "name": "7-17-2026_07-32PM_Utah Yaks x Mexico.csv"},
    {"id": "1gJVNviZP3Nk3SRAHuisbnSt4Nu463u7R", "name": "7-17-2026_04-12PM_Anchors 2026 x Red Hots 2026.csv"},
    {"id": "1ts3wzyBkUmWHxdtEUd21UOBvymPSO-Yb", "name": "7-16-2026_11-58AM_Puerto Rico x Royals 2026.csv"},
    {"id": "1JZL_ZBkJjmxi2oV0r5joar_dSnrtnNLe", "name": "7-17-2026_12-38PM_Royals 2026 x Blue Crew 2026.csv"},
    {"id": "1jqh_SFCV28hMz99k18MqhGEaKZluwZ2u", "name": "7-16-2026_05-03PM_Mexico x Anchors 2026.csv"},
    {"id": "1auv4HMpFtFKPchw3JOF7yenNE4NyubHC", "name": "7-16-2026_08-39AM_Utah Yaks x Red Hots 2026.csv"},
    {"id": "1qoger5mG_U9rMWytwElfPa2ke3ij0ZJI", "name": "7-16-2026_02-50PM_Blue Crew 2026 x Aruba.csv"},
    {"id": "1V-W408aXAK1LaYHZeHcRTKUpgrbGoxkA", "name": "7-14-2026_12-48PM_Red Hots 2026 x Mexico.csv"}
]

@st.cache_data(ttl=3600, show_spinner=False)
def load_all_game_telemetry():
    frames = []
    for item in GAME_FILES:
        url = f"https://drive.google.com/uc?export=download&id={item['id']}"
        try:
            res = requests.get(url)
            if res.status_code == 200:
                df = pd.read_csv(io.StringIO(res.content.decode('utf-8', errors='ignore')), low_memory=False)
                if df.empty:
                    continue

                # Multi-season year determination
                if 'Date' in df.columns and df['Date'].dropna().size > 0:
                    df['ParsedDate'] = pd.to_datetime(df['Date'], errors='coerce')
                    df['Season_Year'] = df['ParsedDate'].dt.year.fillna(2026).astype(int)
                else:
                    match = re.search(r"202\d", item['name'])
                    df['Season_Year'] = int(match.group(0)) if match else 2026

                df['Game_Source'] = item['name']

                # Filter tracking noise
                if 'RelSpeed' in df.columns:
                    df = df[(df['RelSpeed'] >= 35.0) & (df['RelSpeed'] <= 106.0)]
                if 'isOutlier' in df.columns:
                    df = df[df['isOutlier'] != True]

                frames.append(df)
        except Exception:
            continue

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

# ----------------- PRO BRANDING & HEADER -----------------
st.title("⚡ Marshalls League Data Engine")
st.markdown("##### **Created by Jordan Jones** | *Next-Gen Ball Flight Kinematics & Player Development System*")

with st.spinner("Streaming TrackMan & WIN Reality telemetry..."):
    data = load_all_game_telemetry()

if data.empty:
    st.error("No telemetry data could be loaded. Please ensure internet access is reachable.")
    st.stop()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🎯 Data Control Room")

# Multi-Season Filter
available_years = sorted(data['Season_Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Season Filter", options=available_years, default=available_years)
df_filtered = data[data['Season_Year'].isin(selected_years)]

# Pitch Types
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
tab_flight, tab_zone, tab_kinematic = st.tabs([
    "🎯 Ball Flight & Pitch Shapes",
    "📐 Plate Distribution & Zone Matrix",
    "⚡ Kinetic Chain & Energy Transfer"
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
            title="Strike Zone Heat & Location Matrix (Catcher Perspective)",
            labels={"PlateLocSide": "Horizontal Location (ft)", "PlateLocHeight": "Vertical Location (ft)"},
            template="plotly_dark"
        )
        fig_zone.add_shape(type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
                           line=dict(color="#00FFCC", width=3))
        fig_zone.update_xaxes(range=[-2.5, 2.5])
        fig_zone.update_yaxes(range=[0, 5])
        st.plotly_chart(fig_zone, use_container_width=True)

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
                    "PelvisMaxAngularVelocity": "Pelvis Max Angular Vel (deg/s)",
                    "ShoulderMaxAngularVelocity": "Torso Max Angular Vel (deg/s)"
                },
                template="plotly_dark"
            )
            st.plotly_chart(fig_kin, use_container_width=True)
