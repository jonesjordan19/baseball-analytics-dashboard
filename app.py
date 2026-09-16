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

# Session state initialization
if "nav_radio" not in st.session_state:
    st.session_state["nav_radio"] = "🏆 League Leaderboard Hub"
if "selected_player" not in st.session_state:
    st.session_state["selected_player"] = None

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
        if 'InducedVertBreak' in combined.columns:
            combined = combined[(combined['InducedVertBreak'] >= -35) & (combined['InducedVertBreak'] <= 35)]
        if 'HorzBreak' in combined.columns:
            combined = combined[(combined['HorzBreak'] >= -35) & (combined['HorzBreak'] <= 35)]
        return combined
    return pd.DataFrame()

# =====================================================================
# PITCH CLASSIFICATION & SEQUENCING ENGINE
# =====================================================================
PITCH_FAMILY_MAP = {
    'Fastball': 'Fastballs',
    'FourSeamFastball': 'Fastballs',
    'TwoSeamFastball': 'Fastballs',
    'Sinker': 'Fastballs',
    'Cutter': 'Fastballs',
    'Slider': 'Breaking',
    'Curveball': 'Breaking',
    'Sweeper': 'Breaking',
    'Slurve': 'Breaking',
    'KnuckleCurve': 'Breaking',
    'Changeup': 'Offspeed',
    'Splitter': 'Offspeed',
    'Forkball': 'Offspeed'
}

PITCH_FAMILY_COLORS = {
    'Fastballs': '#EF4444',  # Red
    'Breaking': '#06B6D4',   # Cyan
    'Offspeed': '#10B981',   # Green
    'UD': '#6B7280'          # Gray (Undefined)
}

def enrich_pitch_sequencing_data(df_input):
    """Computes previous pitch type and previous pitch outcome within plate appearances."""
    if df_input.empty:
        return df_input

    df_seq = df_input.copy()
    sort_cols = [c for c in ['Game_Source', 'Inning', 'PAofInning', 'PitchofPA'] if c in df_seq.columns]
    if sort_cols:
        df_seq = df_seq.sort_values(by=sort_cols).reset_index(drop=True)

    # Classify pitch families
    df_seq['PitchFamily'] = df_seq['TaggedPitchType'].map(PITCH_FAMILY_MAP).fillna('UD')

    # Group by Game and PA to determine relative pitch sequencing
    pa_group_cols = [c for c in ['Game_Source', 'Inning', 'PAofInning'] if c in df_seq.columns]
    if not pa_group_cols:
        pa_group_cols = ['Game_Source']

    # Previous pitch family in the PA
    df_seq['PrevPitchFamily'] = df_seq.groupby(pa_group_cols)['PitchFamily'].shift(1)
    df_seq['PrevPitchCall'] = df_seq.groupby(pa_group_cols)['PitchCall'].shift(1)
    
    # 1. Pitch Sequencing Categories
    def get_pitch_seq_cat(row):
        p_num = row.get('PitchofPA', 1)
        if p_num == 1 or pd.isna(row.get('PrevPitchFamily')):
            return '1P'
        prev = row['PrevPitchFamily']
        if prev == 'Fastballs': return 'AFB'
        if prev == 'Breaking': return 'ABR'
        if prev == 'Offspeed': return 'AOFF'
        return 'AU'

    # 2. Count Sequencing Categories
    def get_count_seq_cat(row):
        b = int(row.get('Balls', 0))
        s = int(row.get('Strikes', 0))
        if b == 0 and s == 0: return '1P'
        if b == 3 and s == 2: return 'Full'
        if s > b: return 'Ahead'
        if b > s: return 'Behind'
        if b == s: return 'Even'
        return 'Other'

    # 3. Result Sequencing Categories (based on outcome of previous pitch)
    def get_result_seq_cat(row):
        p_num = row.get('PitchofPA', 1)
        if p_num == 1 or pd.isna(row.get('PrevPitchCall')):
            return None
        call = str(row['PrevPitchCall']).lower()
        if 'ball' in call or 'hitby' in call:
            return 'AB'   # After Ball
        elif 'strikecalled' in call:
            return 'ACS'  # After Called Strike
        elif 'swinging' in call:
            return 'AW'   # After Whiff
        elif 'foul' in call:
            return 'AF'   # After Foul
        return None

    df_seq['PitchSeqBucket'] = df_seq.apply(get_pitch_seq_cat, axis=1)
    df_seq['CountSeqBucket'] = df_seq.apply(get_count_seq_cat, axis=1)
    df_seq['ResultSeqBucket'] = df_seq.apply(get_result_seq_cat, axis=1)

    return df_seq

def render_100pct_stacked_bar(df_data, category_col, fixed_order, title):
    """Draws an exact 100% horizontal stacked bar chart matching the WIN Reality panel."""
    if df_data.empty or category_col not in df_data.columns:
        return go.Figure()

    ctab = pd.crosstab(df_data[category_col], df_data['PitchFamily'], normalize='index').multiply(100)
    for col in ['Fastballs', 'Breaking', 'Offspeed', 'UD']:
        if col not in ctab.columns:
            ctab[col] = 0.0

    # Retain fixed chronological baseball ordering (bottom-to-top display reversed for standard Y-axis)
    valid_order = [cat for cat in fixed_order if cat in ctab.index]
    ctab = ctab.reindex(valid_order).fillna(0.0)

    fig = go.Figure()
    for family in ['Fastballs', 'Breaking', 'Offspeed', 'UD']:
        fig.add_trace(go.Bar(
            y=ctab.index,
            x=ctab[family],
            name=family,
            orientation='h',
            marker=dict(color=PITCH_FAMILY_COLORS[family]),
            hoverinfo='text',
            hovertext=[f"{y} | {family}: {val:.1f}%" for y, val in zip(ctab.index, ctab[family])]
        ))

    fig.update_layout(
        title=dict(text=f"<b>{title}</b>", font=dict(size=15, color="#111827")),
        barmode='stack',
        height=320,
        margin=dict(l=10, r=15, t=40, b=30),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0, font=dict(size=10)),
        xaxis=dict(range=[0, 100], ticksuffix="%", dtick=25, gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=12, color="#111827", family="Arial Black")),
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF"
    )
    return fig

def render_strike_zone_figure(df_pitches):
    fig = go.Figure()
    fig.add_shape(
        type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
        fillcolor="rgba(0, 150, 255, 0.08)",
        line=dict(color="#111827", width=3)
    )
    x_third = 1.66 / 3.0
    y_third = 2.0 / 3.0
    fig.add_shape(type="line", x0=-0.83 + x_third, y0=1.5, x1=-0.83 + x_third, y1=3.5,
                  line=dict(color="rgba(50,50,50,0.4)", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=0.83 - x_third, y0=1.5, x1=0.83 - x_third, y1=3.5,
                  line=dict(color="rgba(50,50,50,0.4)", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=-0.83, y0=1.5 + y_third, x1=0.83, y1=1.5 + y_third,
                  line=dict(color="rgba(50,50,50,0.4)", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=-0.83, y0=3.5 - y_third, x1=0.83, y1=3.5 - y_third,
                  line=dict(color="rgba(50,50,50,0.4)", width=1.5, dash="dot"))

    fig.add_trace(go.Scatter(
        x=[-0.708, 0.708, 0.708, 0.0, -0.708, -0.708],
        y=[0.6, 0.6, 0.45, 0.25, 0.45, 0.6],
        fill="toself", fillcolor="rgba(180, 185, 195, 0.7)",
        line=dict(color="#111827", width=2),
        mode="lines", showlegend=False, hoverinfo="skip"
    ))

    non_contact = df_pitches[df_pitches['ExitSpeed'].isna() | (df_pitches['ExitSpeed'] < 40)]
    for ptype, group in non_contact.groupby('TaggedPitchType'):
        hover_info = group.apply(lambda r: f"{ptype} | {r.get('RelSpeed', 0):.1f} mph<br>Count: {r.get('Balls', 0)}-{r.get('Strikes', 0)}<br>Call: {r.get('PitchCall', '')}", axis=1)
        fig.add_trace(go.Scatter(
            x=group['PlateLocSide'], y=group['PlateLocHeight'],
            mode='markers', name=ptype,
            hovertext=hover_info, hoverinfo="text",
            marker=dict(size=10, opacity=0.85)
        ))

    contact_p = df_pitches[df_pitches['ExitSpeed'].notna() & (df_pitches['ExitSpeed'] >= 40)]
    if not contact_p.empty:
        hover_contact = contact_p.apply(lambda r: f"CONTACT: {r.get('TaggedPitchType', '')} | {r.get('RelSpeed', 0):.1f} mph<br>EV: {r.get('ExitSpeed', 0):.1f} mph | LA: {r.get('Angle', 0):.0f}°<br>Result: {r.get('PlayResult', '')}", axis=1)
        fig.add_trace(go.Scatter(
            x=contact_p['PlateLocSide'], y=contact_p['PlateLocHeight'],
            mode='markers', name="In Play",
            hovertext=hover_contact, hoverinfo="text",
            marker=dict(size=16, color='rgba(0,0,0,0)', line=dict(color='#EAB308', width=3.5))
        ))

    fig.update_xaxes(range=[-2.2, 2.2], title="Horizontal Plate Location (ft)", zeroline=False, gridcolor="rgba(0,0,0,0.06)")
    fig.update_yaxes(range=[0.0, 4.5], title="Height from Ground (ft)", zeroline=False, gridcolor="rgba(0,0,0,0.06)")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                      plot_bgcolor="rgba(245, 247, 250, 0.6)")
    return fig

def render_field_spray_chart(batted_df):
    fig = go.Figure()
    rad_lf = np.radians(135)
    rad_rf = np.radians(45)
    lf_x, lf_y = 330 * np.cos(rad_lf), 330 * np.sin(rad_lf)
    rf_x, rf_y = 330 * np.cos(rad_rf), 330 * np.sin(rad_rf)

    fig.add_trace(go.Scatter(x=[0, lf_x], y=[0, lf_y], mode='lines', line=dict(color="rgba(100,100,100,0.7)", width=2), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[0, rf_x], y=[0, rf_y], mode='lines', line=dict(color="rgba(100,100,100,0.7)", width=2), showlegend=False, hoverinfo='skip'))

    angles = np.linspace(45, 135, 60)
    radii = 330 + 70 * np.sin(np.radians(angles - 45) * 2)
    arc_x = radii * np.cos(np.radians(angles))
    arc_y = radii * np.sin(np.radians(angles))
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines', line=dict(color="rgba(80,80,80,0.8)", width=3), showlegend=False, hoverinfo='skip'))

    b1_x, b1_y = 90 * np.cos(np.radians(45)), 90 * np.sin(np.radians(45))
    b2_x, b2_y = 0, 127.28
    b3_x, b3_y = 90 * np.cos(np.radians(135)), 90 * np.sin(np.radians(135))
    fig.add_trace(go.Scatter(
        x=[0, b1_x, b2_x, b3_x, 0], y=[0, b1_y, b2_y, b3_y, 0],
        mode='lines', line=dict(color="rgba(120,120,120,0.5)", width=1.5, dash="dash"),
        showlegend=False, hoverinfo='skip'
    ))

    if not batted_df.empty and 'Direction' in batted_df.columns and 'Distance' in batted_df.columns:
        valid_bip = batted_df[batted_df['Distance'].notna() & (batted_df['Distance'] > 15)].copy()
        if not valid_bip.empty:
            theta_rad = np.radians(90 - valid_bip['Direction'])
            valid_bip['Field_X'] = valid_bip['Distance'] * np.cos(theta_rad)
            valid_bip['Field_Y'] = valid_bip['Distance'] * np.sin(theta_rad)

            hover_text = valid_bip.apply(
                lambda r: f"EV: {r.get('ExitSpeed', 0):.1f} mph<br>Dist: {r.get('Distance', 0):.0f} ft<br>LA: {r.get('Angle', 0):.0f}°<br>Pitch: {r.get('TaggedPitchType', '')}<br>Result: {r.get('PlayResult', '')}",
                axis=1
            )
            fig.add_trace(go.Scatter(
                x=valid_bip['Field_X'], y=valid_bip['Field_Y'],
                mode='markers',
                marker=dict(
                    size=12, color=valid_bip['ExitSpeed'],
                    colorscale='Turbo', cmin=70, cmax=105,
                    colorbar=dict(title="EV (mph)", x=1.02, thickness=12),
                    line=dict(color='black', width=1)
                ),
                text=hover_text, hoverinfo="text", name="Batted Ball"
            ))

    fig.update_xaxes(range=[-260, 260], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-20, 430], showgrid=False, zeroline=False, visible=False)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="rgba(245, 247, 250, 0.6)")
    return fig

def select_player_callback(player_name, target_view):
    st.session_state["selected_player"] = player_name
    st.session_state["nav_radio"] = target_view

def render_clickable_leaderboard(df_ranked, name_col, metric_label, metric_col, submetric_label, submetric_col, player_type, key_prefix):
    target_view = "🔥 Individual Hitter Card" if player_type == "Hitter" else "🛡️ Individual Pitcher Card"
    for idx, r in df_ranked.iterrows():
        p_name = str(r[name_col])
        m_val = r[metric_col]
        sub_val = r[submetric_col]
        
        c_rank, c_btn, c_stat = st.columns([0.6, 3.2, 2.2])
        c_rank.markdown(f"**#{idx+1}**")
        c_btn.button(
            f"{p_name}",
            key=f"{key_prefix}_{idx}_{p_name}",
            on_click=select_player_callback,
            args=(p_name, target_view),
            use_container_width=True
        )
        c_stat.markdown(f"**{m_val}** {metric_label} <span style='color:gray; font-size:12px;'>({sub_val} {submetric_label})</span>", unsafe_allow_html=True)

# ----------------- BRANDING & HEADER -----------------
st.title("⚡ Marshalls League Data Engine")
st.markdown("##### **Created by Jordan Jones** | *Official WIN Reality SmartPark Analytics & Scouting Suite*")

with st.spinner("Streaming Marshalls League telemetry..."):
    data = load_marshalls_telemetry(MANIFEST_SHEET_ID)

if data.empty:
    st.warning("Telemetry is loading. Please check permissions on the Google Sheet.")
    st.stop()

# ----------------- SIDEBAR CONTROLS -----------------
st.sidebar.header("🎯 Navigation & Controls")

report_options = [
    "🏆 League Leaderboard Hub",
    "🔥 Individual Hitter Card",
    "🛡️ Individual Pitcher Card",
    "📊 Team Game Summary"
]

report_scope = st.sidebar.radio(
    "Navigation View",
    report_options,
    key="nav_radio"
)

available_years = sorted(data['Season_Year'].dropna().unique())
selected_year = st.sidebar.selectbox("Season Year", options=available_years, index=0)
season_data = data[data['Season_Year'] == selected_year]

# =====================================================================
# VIEW 1: LEAGUE LEADERBOARD HUB
# =====================================================================
if report_scope == "🏆 League Leaderboard Hub":
    st.subheader(f"🏆 Marshalls League Official Leaderboard ({selected_year})")
    st.caption("Click directly on any player's name button to immediately open their full-season scouting card.")

    lb_tab_hit, lb_tab_pitch = st.tabs(["💥 Hitting Leaderboards", "🎯 Pitching Leaderboards"])

    with lb_tab_hit:
        batted_all = season_data[season_data['ExitSpeed'].notna() & (season_data['ExitSpeed'] >= 40) & (season_data['Batter'].notna())]
        hitter_agg = batted_all.groupby('Batter').agg(
            BIP=('ExitSpeed', 'count'),
            Max_EV=('ExitSpeed', 'max'),
            Avg_EV=('ExitSpeed', 'mean'),
            Hard_Hits=('ExitSpeed', lambda x: (x >= 90.0).sum()),
            Sweet_Spot_Hits=('Angle', lambda x: ((x >= 8.0) & (x <= 32.0)).sum()),
            Max_Dist=('Distance', 'max')
        ).reset_index()

        hitter_agg['Hard_Hit_%'] = ((hitter_agg['Hard_Hits'] / hitter_agg['BIP']) * 100).round(1)
        hitter_agg['Sweet_Spot_%'] = ((hitter_agg['Sweet_Spot_Hits'] / hitter_agg['BIP']) * 100).round(1)
        hitter_agg['Avg_EV'] = hitter_agg['Avg_EV'].round(1)
        hitter_agg['Max_EV'] = hitter_agg['Max_EV'].round(1)

        c_h1, c_h2 = st.columns(2)
        with c_h1:
            st.markdown("#### 🚀 **Top 10 Max Exit Velocity (Raw Power)**")
            top_max_ev = hitter_agg.sort_values(by='Max_EV', ascending=False).head(10).reset_index(drop=True)
            render_clickable_leaderboard(top_max_ev, 'Batter', 'mph', 'Max_EV', 'avg', 'Avg_EV', 'Hitter', 'h_max_ev')

            st.write("")
            st.markdown("#### 🎯 **Top 10 Hard-Hit % (90+ mph, min 5 BIP)**")
            top_hh = hitter_agg[hitter_agg['BIP'] >= 5].sort_values(by='Hard_Hit_%', ascending=False).head(10).reset_index(drop=True)
            render_clickable_leaderboard(top_hh, 'Batter', '%', 'Hard_Hit_%', 'batted', 'BIP', 'Hitter', 'h_hh')

        with c_h2:
            st.markdown("#### ⚡ **Top 10 Average Exit Velocity (min 5 BIP)**")
            top_avg_ev = hitter_agg[hitter_agg['BIP'] >= 5].sort_values(by='Avg_EV', ascending=False).head(10).reset_index(drop=True)
            render_clickable_leaderboard(top_avg_ev, 'Batter', 'mph', 'Avg_EV', 'max', 'Max_EV', 'Hitter', 'h_avg_ev')

            st.write("")
            st.markdown("#### 📐 **Top 10 Sweet-Spot % (8°-32° LA, min 5 BIP)**")
            top_sw = hitter_agg[hitter_agg['BIP'] >= 5].sort_values(by='Sweet_Spot_%', ascending=False).head(10).reset_index(drop=True)
            render_clickable_leaderboard(top_sw, 'Batter', '%', 'Sweet_Spot_%', 'max', 'Max_EV', 'Hitter', 'h_sw')

    with lb_tab_pitch:
        pitchers_all = season_data[season_data['Pitcher'].notna() & (season_data['Pitcher'] != '')]
        fb_all = pitchers_all[pitchers_all['TaggedPitchType'] == 'Fastball']

        p_fb_agg = fb_all.groupby('Pitcher').agg(
            FB_Pitches=('RelSpeed', 'count'),
            Max_FB=('RelSpeed', 'max'),
            Avg_FB=('RelSpeed', 'mean'),
            Avg_IVB=('InducedVertBreak', 'mean'),
            Avg_Spin=('SpinRate', 'mean')
        ).reset_index()

        p_fb_agg['Max_FB'] = p_fb_agg['Max_FB'].round(1)
        p_fb_agg['Avg_FB'] = p_fb_agg['Avg_FB'].round(1)
        p_fb_agg['Avg_IVB'] = p_fb_agg['Avg_IVB'].round(1)
        p_fb_agg['Avg_Spin'] = p_fb_agg['Avg_Spin'].round(0)

        p_control_agg = pitchers_all.groupby('Pitcher').agg(
            Total_Pitches=('PitchCall', 'count'),
            Strikes=('PitchCall', lambda x: x.astype(str).str.contains("Strike|Foul|InPlay", case=False, na=False).sum()),
            FP_Total=('PitchofPA', lambda x: (x == 1).sum()),
            FP_Strikes=('PitchCall', lambda x: ((pitchers_all.loc[x.index, 'PitchofPA'] == 1) & (x.astype(str).str.contains("Strike|Foul|InPlay", case=False, na=False))).sum())
        ).reset_index()

        p_control_agg['Strike_%'] = ((p_control_agg['Strikes'] / p_control_agg['Total_Pitches']) * 100).round(1)
        p_control_agg['FP_Strike_%'] = ((p_control_agg['FP_Strikes'] / p_control_agg['FP_Total'].replace(0, 1)) * 100).round(1)

        c_p1, c_p2 = st.columns(2)
        with c_p1:
            st.markdown("#### 🔥 **Top 10 Peak Fastball Velocity**")
            top_fb = p_fb_agg.sort_values(by='Max_FB', ascending=False).head(10).reset_index(drop=True)
            render_clickable_leaderboard(top_fb, 'Pitcher', 'mph', 'Max_FB', 'avg', 'Avg_FB', 'Pitcher', 'p_top_fb')

            st.write("")
            st.markdown("#### 🎯 **Top 10 Strike Throwing % (min 30 Pitches)**")
            top_strikes = p_control_agg[p_control_agg['Total_Pitches'] >= 30].sort_values(by='Strike_%', ascending=False).head(10).reset_index(drop=True)
            render_clickable_leaderboard(top_strikes, 'Pitcher', '%', 'Strike_%', 'pitches', 'Total_Pitches', 'Pitcher', 'p_top_strikes')

        with c_p2:
            st.markdown("#### 🌪️ **Top 10 Fastball Ride / IVB (min 15 Fastballs)**")
            top_ivb = p_fb_agg[p_fb_agg['FB_Pitches'] >= 15].sort_values(by='Avg_IVB', ascending=False).head(10).reset_index(drop=True)
            render_clickable_leaderboard(top_ivb, 'Pitcher', 'in', 'Avg_IVB', 'mph', 'Avg_FB', 'Pitcher', 'p_top_ivb')

            st.write("")
            st.markdown("#### 🥊 **Top 10 First-Pitch Strike % (min 10 PAs)**")
            top_fps = p_control_agg[p_control_agg['FP_Total'] >= 10].sort_values(by='FP_Strike_%', ascending=False).head(10).reset_index(drop=True)
            render_clickable_leaderboard(top_fps, 'Pitcher', '%', 'FP_Strike_%', 'faced', 'FP_Total', 'Pitcher', 'p_top_fps')

# =====================================================================
# VIEW 2: INDIVIDUAL HITTER REPORT CARD
# =====================================================================
elif report_scope == "🔥 Individual Hitter Card":
    if st.button("⬅️ Back to League Leaderboard"):
        st.session_state["nav_radio"] = "🏆 League Leaderboard Hub"
        st.rerun()

    batters = sorted([b for b in season_data['Batter'].dropna().unique() if str(b).strip()])
    if not batters:
        st.warning("No hitter data tracked for this selection.")
        st.stop()

    default_batter_idx = 0
    if st.session_state.get("selected_player") in batters:
        default_batter_idx = batters.index(st.session_state["selected_player"])

    selected_batter = st.sidebar.selectbox("Select Batter", options=batters, index=default_batter_idx)
    st.session_state["selected_player"] = selected_batter

    player_games = ["All Games (Season Cumulative)"] + sorted([g for g in season_data[season_data['Batter'] == selected_batter]['Game_Source'].dropna().unique()])
    selected_game = st.sidebar.selectbox("Game Filter", options=player_games, index=0)
    
    b_data = season_data[season_data['Batter'] == selected_batter].copy()
    if selected_game != "All Games (Season Cumulative)":
        b_data = b_data[b_data['Game_Source'] == selected_game]

    # Enrich sequencing features
    b_data = enrich_pitch_sequencing_data(b_data)

    total_pitches = len(b_data)
    swings = len(b_data[b_data['PitchCall'].astype(str).str.contains("StrikeSwinging|Foul|InPlay", case=False, na=False)])
    in_play = b_data[b_data['ExitSpeed'].notna() & (b_data['ExitSpeed'] >= 40)]
    hard_hits = in_play[in_play['ExitSpeed'] >= 90.0]
    sweet_spot = in_play[(in_play['Angle'] >= 8.0) & (in_play['Angle'] <= 32.0)]
    ground_balls = in_play[in_play['Angle'] < 8.0]
    fly_balls = in_play[in_play['Angle'] > 32.0]
    
    avg_ev = in_play['ExitSpeed'].mean() if not in_play.empty else 0
    max_ev = in_play['ExitSpeed'].max() if not in_play.empty else 0
    max_dist = in_play['Distance'].max() if 'Distance' in in_play.columns and in_play['Distance'].notna().any() else 0

    st.subheader(f"Hitter Postgame Report: **{selected_batter}**")
    st.caption(f"Scope: {selected_game} | Season: {selected_year}")

    t1, t2 = st.columns(2)
    with t1:
        if max_ev >= 95:
            st.success(f"**Barrel was loud:** 100+ exit velo recorded ({max_ev:.1f} mph). Pure collegiate power.")
        elif avg_ev >= 88:
            st.success(f"**Consistent contact:** Solid contact quality averaging {avg_ev:.1f} mph off the bat.")
        else:
            st.info("**Working the counts:** Fought into deep counts and saw quality pitches.")

    with t2:
        if len(ground_balls) > len(fly_balls) and len(in_play) > 0:
            st.warning(f"**Pick it up:** {len(ground_balls)} of {len(in_play)} balls in play stayed on the ground. Match the pitch plane and elevate.")
        elif len(sweet_spot) > 0:
            st.success(f"**Good angles:** {len(sweet_spot)} of {len(in_play)} balls in play were squared in the 8°-32° sweet-spot zone.")
        else:
            st.info("**Aggression on strikes:** Attack early count fastballs in the strike zone.")

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Hard-Hit Rate (90+)", f"{len(hard_hits)}/{len(in_play)}" if len(in_play) > 0 else "0/0")
    k2.metric("Average Exit Velo", f"{avg_ev:.1f} mph" if avg_ev > 0 else "N/A")
    k3.metric("Max Exit Velo", f"{max_ev:.1f} mph" if max_ev > 0 else "N/A")
    k4.metric("Max Distance", f"{max_dist:.0f} ft" if max_dist > 0 else "N/A")
    k5.metric("Pitches / Swings", f"{total_pitches} / {swings}")

    st.divider()

    # TWO COLUMNS: STRIKE ZONE & FIELD SPRAY CHART SIDE-BY-SIDE
    col_zone, col_spray = st.columns([1, 1])
    with col_zone:
        st.markdown("#### **Pitches Seen (Catcher's View)**")
        if 'PlateLocSide' in b_data.columns and 'PlateLocHeight' in b_data.columns:
            st.plotly_chart(render_strike_zone_figure(b_data), use_container_width=True)

    with col_spray:
        st.markdown("#### **Field Spray Chart (Batted Ball Trajectories)**")
        st.plotly_chart(render_field_spray_chart(in_play), use_container_width=True)

    st.divider()

    # ----------------- THE 3-CHART PITCH SEQUENCING SUITE -----------------
    st.markdown("### 📊 **Pitch Sequencing Intelligence**")
    st.caption("How opposing pitching staffs attacked this hitter based on previous pitch, count state, and pitch result:")

    seq_c1, seq_c2, seq_c3 = st.columns(3)

    with seq_c1:
        # Chart 1: Pitch Sequencing (1P, ABR, AFB, AU, AOFF)
        order_pseq = ['1P', 'ABR', 'AFB', 'AU', 'AOFF']
        fig_pseq = render_100pct_stacked_bar(b_data, 'PitchSeqBucket', order_pseq, "Pitch Sequencing")
        st.plotly_chart(fig_pseq, use_container_width=True)

    with seq_c2:
        # Chart 2: Count Sequencing (1P, Ahead, Behind, Even, Full)
        order_cseq = ['1P', 'Ahead', 'Behind', 'Even', 'Full']
        fig_cseq = render_100pct_stacked_bar(b_data, 'CountSeqBucket', order_cseq, "Count Sequencing")
        st.plotly_chart(fig_cseq, use_container_width=True)

    with seq_c3:
        # Chart 3: Result Sequencing (AB, ACS, AW, AF)
        order_rseq = ['AB', 'ACS', 'AW', 'AF']
        valid_res_data = b_data[b_data['ResultSeqBucket'].notna()]
        fig_rseq = render_100pct_stacked_bar(valid_res_data, 'ResultSeqBucket', order_rseq, "Result Sequencing")
        st.plotly_chart(fig_rseq, use_container_width=True)

    st.divider()

    # AT-BAT SUMMARY LOG
    st.markdown("#### **At-Bat Summary Log**")
    if not in_play.empty:
        def categorize_trajectory(row):
            la = row.get('Angle', 0)
            if la < 8: return "Ground Ball"
            elif 8 <= la <= 32: return "Line Drive"
            elif 32 < la <= 50: return "Fly Ball"
            return "Pop Up"

        def categorize_direction(row):
            d = row.get('Direction', 0)
            if d < -15: return "Left"
            elif -15 <= d <= 15: return "Center"
            return "Right"

        in_play_display = in_play.copy()
        in_play_display['Contact'] = in_play_display.apply(categorize_trajectory, axis=1)
        in_play_display['Field'] = in_play_display.apply(categorize_direction, axis=1)
        in_play_display['Count'] = in_play_display.apply(lambda r: f"{int(r.get('Balls', 0))}-{int(r.get('Strikes', 0))}", axis=1)
        in_play_display['Exit Velocity'] = in_play_display['ExitSpeed'].round(1).astype(str) + " mph"
        in_play_display['Launch Angle'] = in_play_display['Angle'].round(0).astype(int).astype(str) + "°"
        in_play_display['Distance (ft)'] = in_play_display['Distance'].fillna(0).round(0).astype(int).astype(str) + " ft"

        display_cols = ['Game_Source', 'Inning', 'Count', 'PlayResult', 'Contact', 'Exit Velocity', 'Launch Angle', 'Distance (ft)', 'Field']
        valid_disp_cols = [c for c in display_cols if c in in_play_display.columns]
        
        st.dataframe(
            in_play_display[valid_disp_cols].rename(columns={'Game_Source': 'Game', 'PlayResult': 'Outcome'}),
            use_container_width=True,
            hide_index=True
        )

# =====================================================================
# VIEW 3: INDIVIDUAL PITCHER REPORT CARD
# =====================================================================
elif report_scope == "🛡️ Individual Pitcher Card":
    if st.button("⬅️ Back to League Leaderboard"):
        st.session_state["nav_radio"] = "🏆 League Leaderboard Hub"
        st.rerun()

    pitchers = sorted([p for p in season_data['Pitcher'].dropna().unique() if str(p).strip()])
    if not pitchers:
        st.warning("No pitcher data tracked for this selection.")
        st.stop()

    default_pitcher_idx = 0
    if st.session_state.get("selected_player") in pitchers:
        default_pitcher_idx = pitchers.index(st.session_state["selected_player"])

    selected_pitcher = st.sidebar.selectbox("Select Pitcher", options=pitchers, index=default_pitcher_idx)
    st.session_state["selected_player"] = selected_pitcher

    pitcher_games = ["All Games (Season Cumulative)"] + sorted([g for g in season_data[season_data['Pitcher'] == selected_pitcher]['Game_Source'].dropna().unique()])
    selected_game = st.sidebar.selectbox("Game Filter", options=pitcher_games, index=0)

    p_data = season_data[season_data['Pitcher'] == selected_pitcher].copy()
    if selected_game != "All Games (Season Cumulative)":
        p_data = p_data[p_data['Game_Source'] == selected_game]

    # Enrich sequencing features for pitcher
    p_data = enrich_pitch_sequencing_data(p_data)

    total_p = len(p_data)
    strikes = len(p_data[p_data['PitchCall'].astype(str).str.contains("Strike|Foul|InPlay", case=False, na=False)])
    strike_pct = (strikes / total_p * 100) if total_p > 0 else 0
    
    first_pitches = p_data[p_data['PitchofPA'] == 1]
    fp_strikes = len(first_pitches[first_pitches['PitchCall'].astype(str).str.contains("Strike|Foul|InPlay", case=False, na=False)])
    fp_strike_pct = (fp_strikes / len(first_pitches) * 100) if len(first_pitches) > 0 else 0
    
    fb_df = p_data[p_data['TaggedPitchType'] == 'Fastball']
    avg_fb = fb_df['RelSpeed'].mean() if not fb_df.empty else 0
    max_fb = fb_df['RelSpeed'].max() if not fb_df.empty else 0

    st.subheader(f"Pitcher Postgame Report: **{selected_pitcher}**")
    st.caption(f"Scope: {selected_game} | Season: {selected_year}")

    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if fp_strike_pct >= 65:
            st.success(f"**Pounded the zone early:** {fp_strike_pct:.0f}% first-pitch strikes. Set the tone and dictated counts.")
        else:
            st.warning(f"**Mix pitch one:** First-pitch strike rate was {fp_strike_pct:.0f}%. Get ahead early to open secondary pitches.")
    with col_t2:
        if strike_pct >= 62:
            st.success(f"**In the zone all day:** {strike_pct:.0f}% total strikes. Filled up the zone.")
        else:
            st.info(f"**Count control:** Focus on winning 1-1 and 2-2 counts to avoid deep pitch counts.")

    pk1, pk2, pk3, pk4, pk5 = st.columns(5)
    pk1.metric("Total Pitches", f"{total_p}")
    pk2.metric("Strike %", f"{strike_pct:.0f}%")
    pk3.metric("Avg FB Velo", f"{avg_fb:.1f} mph" if avg_fb > 0 else "N/A")
    pk4.metric("Max FB Velo", f"{max_fb:.1f} mph" if max_fb > 0 else "N/A")
    pk5.metric("First-Pitch Strike %", f"{fp_strike_pct:.0f}%")

    st.divider()

    p_col1, p_col2 = st.columns([1.1, 1.3])
    with p_col1:
        st.markdown("#### **Pitch Movement (Pitcher's View)**")
        fig_mov = px.scatter(
            p_data, x="HorzBreak", y="InducedVertBreak",
            color="TaggedPitchType", hover_data=["RelSpeed", "SpinRate"],
            labels={"HorzBreak": "Horizontal Break (HB) [in]", "InducedVertBreak": "Induced Vertical Break (IVB) [in]"},
            height=400
        )
        fig_mov.update_xaxes(range=[-25, 25], gridcolor="rgba(0,0,0,0.06)")
        fig_mov.update_yaxes(range=[-25, 25], gridcolor="rgba(0,0,0,0.06)")
        fig_mov.add_hline(y=0, line_dash="dash", line_color="#888888")
        fig_mov.add_vline(x=0, line_dash="dash", line_color="#888888")
        fig_mov.update_layout(plot_bgcolor="rgba(245, 247, 250, 0.6)")
        st.plotly_chart(fig_mov, use_container_width=True)

    with p_col2:
        st.markdown("#### **Location & Damage Allowed (Catcher's View)**")
        if 'PlateLocSide' in p_data.columns and 'PlateLocHeight' in p_data.columns:
            st.plotly_chart(render_strike_zone_figure(p_data), use_container_width=True)

    st.divider()

    # ----------------- THE 3-CHART PITCH SEQUENCING SUITE FOR PITCHER -----------------
    st.markdown("### 📊 **Pitcher Arsenal Sequencing Tendencies**")
    st.caption("Usage breakdown by previous pitch, count state, and previous pitch result:")

    p_seq_c1, p_seq_c2, p_seq_c3 = st.columns(3)

    with p_seq_c1:
        order_pseq = ['1P', 'ABR', 'AFB', 'AU', 'AOFF']
        fig_p_pseq = render_100pct_stacked_bar(p_data, 'PitchSeqBucket', order_pseq, "Pitch Sequencing")
        st.plotly_chart(fig_p_pseq, use_container_width=True)

    with p_seq_c2:
        order_cseq = ['1P', 'Ahead', 'Behind', 'Even', 'Full']
        fig_p_cseq = render_100pct_stacked_bar(p_data, 'CountSeqBucket', order_cseq, "Count Sequencing")
        st.plotly_chart(fig_p_cseq, use_container_width=True)

    with p_seq_c3:
        order_rseq = ['AB', 'ACS', 'AW', 'AF']
        p_valid_res = p_data[p_data['ResultSeqBucket'].notna()]
        fig_p_rseq = render_100pct_stacked_bar(p_valid_res, 'ResultSeqBucket', order_rseq, "Result Sequencing")
        st.plotly_chart(fig_p_rseq, use_container_width=True)

# =====================================================================
# VIEW 4: TEAM GAME SUMMARY
# =====================================================================
else:
    games = ["All Games (Season Cumulative)"] + sorted([g for g in season_data['Game_Source'].dropna().unique()])
    selected_game = st.sidebar.selectbox("Game Select", options=games)
    active_data = season_data[season_data['Game_Source'] == selected_game] if selected_game != "All Games (Season Cumulative)" else season_data

    st.subheader(f"Team Postgame Benchmark Report")
    st.caption(f"Game: {selected_game} | Season: {selected_year}")

    batted_team = active_data[active_data['ExitSpeed'].notna() & (active_data['ExitSpeed'] >= 40)]
    hard_hits_team = len(batted_team[batted_team['ExitSpeed'] >= 90.0])
    sweet_spot_team = len(batted_team[(batted_team['Angle'] >= 8.0) & (batted_team['Angle'] <= 32.0)])
    
    hh_rate = (hard_hits_team / len(batted_team) * 100) if len(batted_team) > 0 else 0
    sw_rate = (sweet_spot_team / len(batted_team) * 100) if len(batted_team) > 0 else 0

    st.markdown("#### **Team Performance vs. College Targets**")
    t1, t2 = st.columns(2)
    with t1:
        st.write(f"**Hard-Hit Rate (90+ MPH): {hh_rate:.0f}%** *(Target: 40%+)*")
        st.progress(min(int(hh_rate) / 100, 1.0))
        st.write(f"**Launch Angle Sweet-Spot % (8°-32°): {sw_rate:.0f}%** *(Target: 35%+)*")
        st.progress(min(int(sw_rate) / 100, 1.0))

    with t2:
        gb_count = len(batted_team[batted_team['Angle'] < 8])
        ld_count = len(batted_team[(batted_team['Angle'] >= 8) & (batted_team['Angle'] <= 32)])
        total_bip = len(batted_team)
        st.write(f"**Line-Drive %: {(ld_count/total_bip*100):.0f}%** *(Target: 25%+)*" if total_bip > 0 else "Line Drive %: N/A")
        st.progress(min((ld_count/total_bip) if total_bip > 0 else 0, 1.0))
        st.write(f"**Ground-Ball %: {(gb_count/total_bip*100):.0f}%** *(Target: Keep under 40%)*" if total_bip > 0 else "Ground Ball %: N/A")
        st.progress(min((gb_count/total_bip) if total_bip > 0 else 0, 1.0))

    st.divider()

    st.markdown("#### **Hitter Leaderboard**")
    if 'Batter' in active_data.columns:
        leaderboard = active_data[active_data['ExitSpeed'].notna()].groupby('Batter').agg({
            'ExitSpeed': ['count', 'mean', 'max'],
            'Angle': lambda x: (len(x[(x >= 8) & (x <= 32)]) / len(x) * 100) if len(x) > 0 else 0
        }).round(1)
        leaderboard.columns = ['Balls in Play', 'Avg EV', 'Max EV', 'Sweet Spot %']
        st.dataframe(leaderboard.reset_index().sort_values(by='Max EV', ascending=False), use_container_width=True, hide_index=True)
