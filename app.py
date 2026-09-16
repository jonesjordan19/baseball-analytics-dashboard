import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import re
from io import StringIO

st.set_page_config(page_title="Pro Baseball Sabermetrics Hub", layout="wide")

FOLDER_ID = "1aJlhryPy5pPqEcbt-EvkEIiLrGwuQtnd"

@st.cache_data(ttl=600)  # Caches data for 10 minutes, then re-checks Drive
def fetch_and_process_game_data(folder_id):
    """Fetches CSVs from public Google Drive folder and parses metrics."""
    # Discover files via Google Drive web listing
    folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
    resp = requests.get(folder_url)
    
    # Extract file IDs and titles matching CSV files
    file_pattern = re.findall(r'\["([a-zA-Z0-9_-]{28,35})",\["([^"]+\.csv)"', resp.text)
    
    # Fallback to direct download list if regex structure shifts
    discovered_files = list(set(file_pattern))
    
    frames = []
    
    for file_id, file_name in discovered_files:
        try:
            download_url = f"https://drive.google.com/uc?export=download&id={file_id}"
            csv_data = requests.get(download_url).content.decode('utf-8', errors='ignore')
            df = pd.read_csv(StringIO(csv_data), low_memory=False)
            
            if df.empty:
                continue

            # Extract Year from Date column or filename
            if 'Date' in df.columns and df['Date'].dropna().size > 0:
                df['ParsedDate'] = pd.to_datetime(df['Date'], errors='coerce')
                df['Season_Year'] = df['ParsedDate'].dt.year.fillna(2026).astype(int)
            else:
                year_match = re.search(r"202\d", file_name)
                df['Season_Year'] = int(year_match.group(0)) if year_match else 2026

            df['Game_File'] = file_name
            
            # Filter radar/camera noise
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

# ----------------- UI & DASHBOARD -----------------
st.title("⚾ Pro Sabermetrics & Development Dashboard")
st.caption("Live Feed from WIN Reality Smart Parks & TrackMan Tracking System")

with st.spinner("Checking Google Drive folder for season files..."):
    data = fetch_and_process_game_data(FOLDER_ID)

if data.empty:
    st.error("No valid game CSV data could be loaded. Make sure the Google Drive folder link sharing is set to 'Anyone with the link can view'.")
    st.stop()

# Sidebar Controls
st.sidebar.header("🎯 Filter Controls")

# Year Selector
all_years = sorted(data['Season_Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Season Year", options=all_years, default=all_years)

# Filter by Year first
df_filtered = data[data['Season_Year'].isin(selected_years)]

# Pitcher & Pitch Type Selectors
pitchers = sorted([p for p in df_filtered['Pitcher'].dropna().unique() if str(p).strip()])
if pitchers:
    selected_pitcher = st.sidebar.selectbox("Select Pitcher", options=["All Pitchers"] + pitchers)
    if selected_pitcher != "All Pitchers":
        df_filtered = df_filtered[df_filtered['Pitcher'] == selected_pitcher]

pitch_types = sorted([pt for pt in df_filtered['TaggedPitchType'].dropna().unique() if str(pt).strip()])
selected_pitches = st.sidebar.multiselect("Pitch Types", options=pitch_types, default=pitch_types)
df_filtered = df_filtered[df_filtered['TaggedPitchType'].isin(selected_pitches)]

# Key Top-Level KPI Cards
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Pitches Tracked", f"{len(df_filtered):,}")
fb_df = df_filtered[df_filtered['TaggedPitchType'] == 'Fastball']
c2.metric("Avg Fastball Velo", f"{fb_df['RelSpeed'].mean():.1f} mph" if not fb_df.empty else "N/A")
c3.metric("Avg Fastball Spin", f"{fb_df['SpinRate'].mean():.0f} rpm" if not fb_df.empty else "N/A")
c4.metric("Seasons Represented", ", ".join(map(str, selected_years)))

st.divider()

# Tab Navigation
tab1, tab2, tab3 = st.tabs(["Pitch Shapes (IVB vs HB)", "Strike Zone & Locations", "Biomechanical Chain"])

with tab1:
    st.subheader("Pitch Arsenal Movement (TrackMan Flight Profile)")
    col_left, col_right = st.columns([2, 1])
    
    with col_left:
        fig_mov = px.scatter(
            df_filtered,
            x="HorzBreak",
            y="InducedVertBreak",
            color="TaggedPitchType",
            symbol="Season_Year" if len(selected_years) > 1 else None,
            hover_data=["RelSpeed", "SpinRate", "Season_Year"],
            title="Induced Vertical Break (IVB) vs. Horizontal Break (HB)",
            labels={"HorzBreak": "Horizontal Break (in)", "InducedVertBreak": "Induced Vertical Break (in)"}
        )
        fig_mov.add_hline(y=0, line_dash="dash", line_color="gray")
        fig_mov.add_vline(x=0, line_dash="dash", line_color="gray")
        st.plotly_chart(fig_mov, use_container_width=True)
        
    with col_right:
        st.write("**Arsenal Summary Table**")
        summary_cols = ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'Extension']
        existing_cols = [c for c in summary_cols if c in df_filtered.columns]
        table_summary = df_filtered.groupby(['Season_Year', 'TaggedPitchType'])[existing_cols].mean().reset_index()
        st.dataframe(table_summary.round(1), use_container_width=True)

with tab2:
    st.subheader("Plate Location Matrix")
    if 'PlateLocSide' in df_filtered.columns and 'PlateLocHeight' in df_filtered.columns:
        fig_zone = px.scatter(
            df_filtered,
            x="PlateLocSide",
            y="PlateLocHeight",
            color="TaggedPitchType",
            hover_data=["RelSpeed", "PitchCall"],
            title="Pitch Locations at Home Plate (Catcher's Perspective)"
        )
        # Standard strike zone outline: x: -0.83 to 0.83 ft, y: 1.5 to 3.5 ft
        fig_zone.add_shape(type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
                           line=dict(color="White", width=3))
        fig_zone.update_xaxes(range=[-2.5, 2.5])
        fig_zone.update_yaxes(range=[0, 5])
        st.plotly_chart(fig_zone, use_container_width=True)
    else:
        st.info("Plate location metrics are not populated.")

with tab3:
    st.subheader("WIN Reality Kinetic Chain & Rotational Sequencing")
    bio_cols = ['PelvisMaxAngularVelocity', 'ShoulderMaxAngularVelocity', 'ArmMaxAngularVelocity']
    
    if set(bio_cols).issubset(df_filtered.columns) and df_filtered['PelvisMaxAngularVelocity'].notna().any():
        fig_bio = px.scatter(
            df_filtered,
            x="PelvisMaxAngularVelocity",
            y="ShoulderMaxAngularVelocity",
            color="TaggedPitchType",
            size="RelSpeed",
            hover_data=["Season_Year"],
            title="Rotational Energy Transfer: Pelvis vs. Shoulder Peak Speed (deg/s)",
            labels={"PelvisMaxAngularVelocity": "Pelvis Max Angular Vel (deg/s)", "ShoulderMaxAngularVelocity": "Torso Max Angular Vel (deg/s)"}
        )
        st.plotly_chart(fig_bio, use_container_width=True)
    else:
        st.info("Biomechanical angular velocity metrics are not available for the selected pitches.")
