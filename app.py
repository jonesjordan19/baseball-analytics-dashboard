import streamlit as st
import pandas as pd
import plotly.express as px
import requests
import io
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(
    page_title="Marshalls League Data Engine",
    page_icon="⚾",
    layout="wide"
)

# ----------------- LIVE GOOGLE SHEET MANIFEST -----------------
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
                
                # Sanitize tracking data
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

    # Parallel multi-threaded download for fast ingestion of all 76+ games
    frames = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = executor.map(fetch_single_csv, tasks)
        for res in results:
            if res is not None:
                frames.append(res)

    if frames:
        return pd.concat(frames, ignore_index=True)
    return pd.DataFrame()

# ----------------- BRANDING & HEADER -----------------
st.title("⚡ Marshalls League Data Engine")
st.markdown("##### **Created by Jordan Jones** | *Next-Gen Ball Flight Kinematics & Player Development System*")

with st.spinner("Streaming full multi-game telemetry lake..."):
    data = load_marshalls_telemetry(MANIFEST_SHEET_ID)

if data.empty:
    st.warning("Telemetry is synchronizing. Verify that your Google Sheet has populated rows and 'Anyone with the link can view' is active.")
    if st.button("🔄 Force Resync"):
        st.cache_data.clear()
        st.rerun()
    st.stop()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🎯 Data Control Room")

# Multi-Season Filter
available_years = sorted(data['Season_Year'].dropna().unique())
selected_years = st.sidebar.multiselect("Season Filter", options=available_years, default=available_years)
df_filtered = data[data['Season_Year'].isin(selected_years)]

# Pitcher Selection
if 'Pitcher' in df_filtered.columns:
    pitchers = sorted([p for p in df_filtered['Pitcher'].dropna().unique() if str(p).strip()])
    if pitchers:
        selected_pitcher = st.sidebar.selectbox("Pitcher Profile", options=["All Pitchers"] + pitchers)
        if selected_pitcher != "All Pitchers":
            df_filtered = df_filtered[df_filtered['Pitcher'] == selected_pitcher]

# Arsenal Selection
pitch_types = sorted([pt for pt in df_filtered['TaggedPitchType'].dropna().unique() if str(pt).strip()])
selected_pitches = st.sidebar.multiselect("Pitch Arsenal", options=pitch_types, default=pitch_types)
df_filtered = df_filtered[df_filtered['TaggedPitchType'].isin(selected_pitches)]

# ----------------- EXECUTIVE KPI CARDS -----------------
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Tracked Pitches", f"{len(df_filtered):,}")

fb_df = df_filtered[df_filtered['TaggedPitchType'] == 'Fastball']
avg_velo = fb_df['RelSpeed'].mean() if not fb_df.empty else df_filtered['RelSpeed'].mean()
k2.metric("Peak FB Velo", f"{fb_df['RelSpeed'].max():.1f} mph" if not fb_df.empty else "N/A")
k3.metric("Avg FB Velo", f"{avg_velo:.1f} mph" if pd.notna(avg_velo) else "N/A")

avg_spin = fb_df['SpinRate'].mean() if not fb_df.empty else df_filtered['SpinRate'].mean()
k4.metric("Avg Spin Rate", f"{avg_spin:.0f} RPM" if pd.notna(avg_spin) else "N/A")

avg_ext = df_filtered['Extension'].mean() if 'Extension' in df_filtered.columns else None
k5.metric("Avg Extension", f"{avg_ext:.1f} ft" if pd.notna(avg_ext) else "N/A")

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
            title="Pitch Movement Profile: Induced Vertical Break (IVB) vs. Horizontal Break (HB)",
            labels={"HorzBreak": "Horizontal Break (Inches)", "InducedVertBreak": "Induced Vertical Break (Inches)"},
            template="plotly_dark"
        )
        fig_shape.add_hline(y=0, line_dash="dash", line_color="#888888")
        fig_shape.add_vline(x=0, line_dash="dash", line_color="#888888")
        st.plotly_chart(fig_shape, use_container_width=True)
        
    with c_table:
        st.markdown("#### **Arsenal Metrics Breakdown**")
        metric_cols = ['RelSpeed', 'SpinRate', 'InducedVertBreak', 'HorzBreak', 'Extension']
        valid_cols = [c for c in metric_cols if c in df_filtered.columns]
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
            hover_data=["RelSpeed"],
            title="Strike Zone Heat & Location Matrix (Catcher View)",
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
        st.success("Currently displaying 2026 Inaugural Season data. When 2027 files are populated in the Google Sheet, side-by-side progression charts will populate automatically.")
