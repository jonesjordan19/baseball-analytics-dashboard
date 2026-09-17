import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import requests
import io
import re
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor

st.set_page_config(
    page_title="Marshalls League Data Engine",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ----------------- PROFESSIONAL FRONT-OFFICE STYLING & DESIGN SYSTEM -----------------
st.markdown("""
<style>
    [data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"], button[kind="header"] {
        display: none !important;
    }
    .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
        max-width: 100% !important;
        background-color: #F8FAFC;
    }
    h1, h2, h3, h4, h5 {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
        color: #0F172A;
        font-weight: 700;
    }
    .metric-card {
        background-color: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        margin-bottom: 12px;
    }
    div.stButton > button {
        width: 100% !important;
        text-align: left !important;
        padding: 0.65rem 1rem !important;
        border-radius: 8px !important;
        border: 1px solid #CBD5E1 !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
        font-weight: 600 !important;
        box-shadow: 0 1px 2px rgba(0,0,0,0.04);
        transition: all 0.2s ease !important;
    }
    div.stButton > button:hover {
        border-color: #2563EB !important;
        background-color: #EFF6FF !important;
        color: #1D4ED8 !important;
    }
</style>
""", unsafe_allow_html=True)

MANIFEST_SHEET_ID = "1Xc3lx4ybIfp9R14ROhCWOD1RpnKIhbNYU76dYQUZdow"

if "nav_view" not in st.session_state:
    st.session_state["nav_view"] = "🏆 Leaderboard Hub"
if "selected_player" not in st.session_state:
    st.session_state["selected_player"] = None
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = []

def extract_game_date(game_source):
    match = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", str(game_source))
    if match:
        m, d, y = map(int, match.groups())
        try:
            return date(y, m, d)
        except Exception:
            pass
    return date(2026, 6, 1)

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
                df['ParsedDate'] = extract_game_date(game_name)
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

PITCH_FAMILY_MAP = {
    'Fastball': 'Fastballs', 'FourSeamFastball': 'Fastballs', 'TwoSeamFastball': 'Fastballs',
    'Sinker': 'Fastballs', 'Cutter': 'Fastballs', 'Slider': 'Breaking', 'Curveball': 'Breaking',
    'Sweeper': 'Breaking', 'Slurve': 'Breaking', 'KnuckleCurve': 'Breaking',
    'Changeup': 'Offspeed', 'Splitter': 'Offspeed', 'Forkball': 'Offspeed'
}

PITCH_FAMILY_COLORS = {
    'Fastballs': '#EF4444', 'Breaking': '#06B6D4', 'Offspeed': '#10B981', 'UD': '#6B7280'
}

def display_shared_sequencing_legend():
    st.markdown("""
        <div style='display: flex; flex-wrap: wrap; gap: 16px; align-items: center; margin-bottom: 8px;'>
            <span style='display: flex; align-items: center;'><span style='height: 10px; width: 10px; background-color: #EF4444; border-radius: 50%; display: inline-block; margin-right: 5px;'></span><small><b>Fastballs</b></small></span>
            <span style='display: flex; align-items: center;'><span style='height: 10px; width: 10px; background-color: #06B6D4; border-radius: 50%; display: inline-block; margin-right: 5px;'></span><small><b>Breaking</b></small></span>
            <span style='display: flex; align-items: center;'><span style='height: 10px; width: 10px; background-color: #10B981; border-radius: 50%; display: inline-block; margin-right: 5px;'></span><small><b>Offspeed</b></small></span>
            <span style='display: flex; align-items: center;'><span style='height: 10px; width: 10px; background-color: #6B7280; border-radius: 50%; display: inline-block; margin-right: 5px;'></span><small><b>UD</b></small></span>
        </div>
    """, unsafe_allow_html=True)

def enrich_pitch_sequencing_data(df_input):
    if df_input.empty:
        return df_input

    df_seq = df_input.copy()
    sort_cols = [c for c in ['Game_Source', 'Inning', 'PAofInning', 'PitchofPA'] if c in df_seq.columns]
    if sort_cols:
        df_seq = df_seq.sort_values(by=sort_cols).reset_index(drop=True)

    df_seq['PitchFamily'] = df_seq['TaggedPitchType'].map(PITCH_FAMILY_MAP).fillna('UD')
    pa_group_cols = [c for c in ['Game_Source', 'Inning', 'PAofInning'] if c in df_seq.columns]
    if not pa_group_cols:
        pa_group_cols = ['Game_Source']

    df_seq['PrevPitchFamily'] = df_seq.groupby(pa_group_cols)['PitchFamily'].shift(1)
    df_seq['PrevPitchCall'] = df_seq.groupby(pa_group_cols)['PitchCall'].shift(1)
    
    def get_pitch_seq_cat(row):
        p_num = row.get('PitchofPA', 1)
        if p_num == 1 or pd.isna(row.get('PrevPitchFamily')):
            return '1P'
        prev = row['PrevPitchFamily']
        if prev == 'Fastballs': return 'AFB'
        if prev == 'Breaking': return 'ABR'
        if prev == 'Offspeed': return 'AOFF'
        return 'AU'

    def get_count_seq_cat(row):
        b = int(row.get('Balls', 0))
        s = int(row.get('Strikes', 0))
        if b == 0 and s == 0: return '1P'
        if b == 3 and s == 2: return 'Full'
        if s > b: return 'Ahead'
        if b > s: return 'Behind'
        if b == s: return 'Even'
        return 'Other'

    def get_result_seq_cat(row):
        p_num = row.get('PitchofPA', 1)
        if p_num == 1 or pd.isna(row.get('PrevPitchCall')):
            return None
        call = str(row['PrevPitchCall']).lower()
        if 'ball' in call or 'hitby' in call: return 'AB'
        elif 'strikecalled' in call: return 'ACS'
        elif 'swinging' in call: return 'AW'
        elif 'foul' in call: return 'AF'
        return None

    df_seq['PitchSeqBucket'] = df_seq.apply(get_pitch_seq_cat, axis=1)
    df_seq['CountSeqBucket'] = df_seq.apply(get_count_seq_cat, axis=1)
    df_seq['ResultSeqBucket'] = df_seq.apply(get_result_seq_cat, axis=1)
    return df_seq

def render_100pct_stacked_bar(df_data, category_col, fixed_order, show_legend=False):
    if df_data.empty or category_col not in df_data.columns:
        return go.Figure()

    ctab = pd.crosstab(df_data[category_col], df_data['PitchFamily'], normalize='index').multiply(100)
    for col in ['Fastballs', 'Breaking', 'Offspeed', 'UD']:
        if col not in ctab.columns:
            ctab[col] = 0.0

    valid_order = [cat for cat in fixed_order if cat in ctab.index]
    ctab = ctab.reindex(valid_order).fillna(0.0)

    fig = go.Figure()
    for family in ['Fastballs', 'Breaking', 'Offspeed', 'UD']:
        fig.add_trace(go.Bar(
            y=ctab.index, x=ctab[family], name=family, orientation='h',
            marker=dict(color=PITCH_FAMILY_COLORS[family]), showlegend=show_legend,
            hoverinfo='text', hovertext=[f"{y} | {family}: {val:.1f}%" for y, val in zip(ctab.index, ctab[family])]
        ))

    fig.update_layout(
        barmode='stack', height=280, margin=dict(l=10, r=15, t=10, b=25),
        xaxis=dict(range=[0, 100], ticksuffix="%", dtick=25, gridcolor="rgba(0,0,0,0.08)", zeroline=False),
        yaxis=dict(autorange="reversed", tickfont=dict(size=11, color="#111827", family="Arial Black")),
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF"
    )
    return fig

def render_strike_zone_figure(df_pitches, title_suffix=""):
    fig = go.Figure()
    fig.add_shape(
        type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
        fillcolor="rgba(0, 150, 255, 0.06)", line=dict(color="#111827", width=3)
    )
    x_third = 1.66 / 3.0
    y_third = 2.0 / 3.0
    fig.add_shape(type="line", x0=-0.83 + x_third, y0=1.5, x1=-0.83 + x_third, y1=3.5, line=dict(color="rgba(50,50,50,0.3)", width=1, dash="dot"))
    fig.add_shape(type="line", x0=0.83 - x_third, y0=1.5, x1=0.83 - x_third, y1=3.5, line=dict(color="rgba(50,50,50,0.3)", width=1, dash="dot"))
    fig.add_shape(type="line", x0=-0.83, y0=1.5 + y_third, x1=0.83, y1=1.5 + y_third, line=dict(color="rgba(50,50,50,0.3)", width=1, dash="dot"))
    fig.add_shape(type="line", x0=-0.83, y0=3.5 - y_third, x1=0.83, y1=3.5 - y_third, line=dict(color="rgba(50,50,50,0.3)", width=1, dash="dot"))

    fig.add_trace(go.Scatter(
        x=[-0.708, 0.708, 0.708, 0.0, -0.708, -0.708], y=[0.6, 0.6, 0.45, 0.25, 0.45, 0.6],
        fill="toself", fillcolor="rgba(180, 185, 195, 0.6)", line=dict(color="#111827", width=2),
        mode="lines", showlegend=False, hoverinfo="skip"
    ))

    if not df_pitches.empty:
        non_contact = df_pitches[df_pitches['ExitSpeed'].isna() | (df_pitches['ExitSpeed'] < 40)]
        for ptype, group in non_contact.groupby('TaggedPitchType'):
            hover_info = group.apply(lambda r: f"{ptype} | {r.get('RelSpeed', 0):.1f} mph<br>Count: {r.get('Balls', 0)}-{r.get('Strikes', 0)}", axis=1)
            fig.add_trace(go.Scatter(
                x=group['PlateLocSide'], y=group['PlateLocHeight'],
                mode='markers', name=ptype, hovertext=hover_info, hoverinfo="text",
                marker=dict(size=8, opacity=0.8)
            ))

        contact_p = df_pitches[df_pitches['ExitSpeed'].notna() & (df_pitches['ExitSpeed'] >= 40)]
        if not contact_p.empty:
            hover_contact = contact_p.apply(lambda r: f"EV: {r.get('ExitSpeed', 0):.1f} mph | LA: {r.get('Angle', 0):.0f}°<br>Result: {r.get('PlayResult', '')}", axis=1)
            fig.add_trace(go.Scatter(
                x=contact_p['PlateLocSide'], y=contact_p['PlateLocHeight'],
                mode='markers', name="In Play", hovertext=hover_contact, hoverinfo="text",
                marker=dict(size=12, color=contact_p['ExitSpeed'], colorscale='Turbo', cmin=70, cmax=105, showscale=False)
            ))

    fig.update_xaxes(range=[-2.2, 2.2], zeroline=False, visible=False)
    fig.update_yaxes(range=[0.0, 4.5], zeroline=False, visible=False)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(245, 247, 250, 0.8)")
    return fig

def render_field_spray_chart(batted_df):
    fig = go.Figure()
    rad_lf, rad_rf = np.radians(135), np.radians(45)
    lf_x, lf_y = 330 * np.cos(rad_lf), 330 * np.sin(rad_lf)
    rf_x, rf_y = 330 * np.cos(rad_rf), 330 * np.sin(rad_rf)

    fig.add_trace(go.Scatter(x=[0, lf_x], y=[0, lf_y], mode='lines', line=dict(color="rgba(100,100,100,0.6)", width=2), showlegend=False, hoverinfo='skip'))
    fig.add_trace(go.Scatter(x=[0, rf_x], y=[0, rf_y], mode='lines', line=dict(color="rgba(100,100,100,0.6)", width=2), showlegend=False, hoverinfo='skip'))

    angles = np.linspace(45, 135, 60)
    radii = 330 + 70 * np.sin(np.radians(angles - 45) * 2)
    arc_x = radii * np.cos(np.radians(angles))
    arc_y = radii * np.sin(np.radians(angles))
    fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode='lines', line=dict(color="rgba(80,80,80,0.7)", width=2.5), showlegend=False, hoverinfo='skip'))

    b1_x, b1_y = 90 * np.cos(np.radians(45)), 90 * np.sin(np.radians(45))
    b2_x, b2_y = 0, 127.28
    b3_x, b3_y = 90 * np.cos(np.radians(135)), 90 * np.sin(np.radians(135))
    fig.add_trace(go.Scatter(
        x=[0, b1_x, b2_x, b3_x, 0], y=[0, b1_y, b2_y, b3_y, 0],
        mode='lines', line=dict(color="rgba(120,120,120,0.4)", width=1.5, dash="dash"),
        showlegend=False, hoverinfo='skip'
    ))

    if not batted_df.empty and 'Direction' in batted_df.columns and 'Distance' in batted_df.columns:
        valid_bip = batted_df[batted_df['Distance'].notna() & (batted_df['Distance'] > 15)].copy()
        if not valid_bip.empty:
            theta_rad = np.radians(90 - valid_bip['Direction'])
            valid_bip['Field_X'] = valid_bip['Distance'] * np.cos(theta_rad)
            valid_bip['Field_Y'] = valid_bip['Distance'] * np.sin(theta_rad)

            hover_text = valid_bip.apply(
                lambda r: f"EV: {r.get('ExitSpeed', 0):.1f} mph<br>Dist: {r.get('Distance', 0):.0f} ft<br>Result: {r.get('PlayResult', '')}",
                axis=1
            )
            fig.add_trace(go.Scatter(
                x=valid_bip['Field_X'], y=valid_bip['Field_Y'], mode='markers',
                marker=dict(size=10, color=valid_bip['ExitSpeed'], colorscale='Turbo', cmin=70, cmax=105, showscale=False),
                text=hover_text, hoverinfo="text", name="Batted Ball"
            ))

    fig.update_xaxes(range=[-260, 260], showgrid=False, zeroline=False, visible=False)
    fig.update_yaxes(range=[-20, 430], showgrid=False, zeroline=False, visible=False)
    fig.update_layout(height=320, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="rgba(245, 247, 250, 0.8)")
    return fig

def render_clickable_leaderboard(df_ranked, name_col, metric_label, metric_col, submetric_label, submetric_col, player_type, key_prefix):
    target_view = "📊 Unified Player Profile"
    for idx, r in df_ranked.iterrows():
        p_name = str(r[name_col])
        m_val = r[metric_col]
        sub_val = r[submetric_col]
        card_label = f"#{idx+1}   {p_name}   —   {m_val} {metric_label}  ({sub_val} {submetric_label})"
        if st.button(card_label, key=f"{key_prefix}_{idx}_{p_name}", use_container_width=True):
            st.session_state["selected_player"] = p_name
            st.session_state["nav_view"] = target_view
            st.rerun()

# ----------------- PROFESSIONAL HEADER -----------------
st.markdown("""
    <div style='display: flex; align-items: center; gap: 14px; margin-bottom: 1rem;'>
        <div style='font-size: 2.5rem;'>🤠</div>
        <div>
            <h1 style='margin: 0; font-size: 1.8rem; color: #0F172A;'>Marshalls League Data Engine</h1>
            <p style='margin: 0; color: #64748B; font-weight: 500;'>Official WIN Reality SmartPark Analytics & Scouting Suite</p>
        </div>
    </div>
""", unsafe_allow_html=True)

with st.spinner("Streaming Marshalls League telemetry..."):
    data = load_marshalls_telemetry(MANIFEST_SHEET_ID)

if data.empty:
    st.warning("Telemetry is loading. Please check permissions on the Google Sheet.")
    st.stop()

if 'ParsedDate' not in data.columns:
    def extract_game_date(game_source):
        match = re.search(r"(\d{1,2})-(\d{1,2})-(\d{4})", str(game_source))
        if match:
            m, d, y = map(int, match.groups())
            try:
                return date(y, m, d)
            except Exception:
                pass
        return date(2026, 6, 1)
    data['ParsedDate'] = data['Game_Source'].apply(extract_game_date)

# ----------------- TOP CONTROLS & SEARCH -----------------
nav_cols = st.columns([1.2, 0.8, 1.4, 1.4])

available_years = sorted(data['Season_Year'].dropna().unique())
selected_year = nav_cols[1].selectbox("Season Year", options=available_years, index=0)
season_raw = data[data['Season_Year'] == selected_year]

if nav_cols[0].button("🏆 Leaderboard Hub", use_container_width=True):
    st.session_state["nav_view"] = "🏆 Leaderboard Hub"
    st.session_state["selected_player"] = None
    st.rerun()

all_players_combined = sorted(list(set(season_raw['Batter'].dropna().unique().tolist() + season_raw['Pitcher'].dropna().unique().tolist())))
search_selection = nav_cols[3].selectbox(
    "Quick Search",
    options=all_players_combined,
    index=None,
    placeholder="🔍 Search any player...",
    label_visibility="collapsed",
    key="universal_player_search"
)

if search_selection:
    st.session_state["selected_player"] = search_selection
    st.session_state["nav_view"] = "📊 Unified Player Profile"
    st.rerun()

all_dates = sorted(season_raw['ParsedDate'].dropna().unique())
min_d = all_dates[0] if all_dates else date(2026, 1, 1)
max_d = all_dates[-1] if all_dates else date(2026, 12, 31)

if "date_filter_range" not in st.session_state:
    st.session_state["date_filter_range"] = (date(2026, 1, 1), date(2026, 12, 31))

date_range = nav_cols[2].date_input(
    "Date Range",
    value=st.session_state["date_filter_range"],
    min_value=date(2026, 1, 1),
    max_value=date(2026, 12, 31),
    label_visibility="collapsed",
    key="date_picker_widget"
)
if date_range != st.session_state["date_filter_range"]:
    st.session_state["date_filter_range"] = date_range

if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    start_d, end_d = date_range
    season_data = season_raw[(season_raw['ParsedDate'] >= start_d) & (season_raw['ParsedDate'] <= end_d)].copy()
else:
    season_data = season_raw.copy()

st.divider()

# =====================================================================
# RESTORED AI SCOUT ASSISTANT CHATBOT
# =====================================================================
with st.expander("🤖 AI Scout Assistant — Ask Anything About Any Player"):
    st.caption("Ask questions like: *'How hard does Ashton Roache hit?'*, *'What is Zaylun Fenn's pitch mix?'*, or *'Who has the highest fastball velo?'*")
    
    for msg in st.session_state["chat_history"][-4:]:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            
    user_prompt = st.chat_input("Ask about a player's velocity, tendencies, pitch mix, or rankings...")
    
    if user_prompt:
        st.session_state["chat_history"].append({"role": "user", "content": user_prompt})
        with st.chat_message("user"):
            st.markdown(user_prompt)
            
        with st.chat_message("assistant"):
            q = user_prompt.lower()
            response_text = ""
            
            gemini_key = st.secrets.get("GEMINI_API_KEY", None)
            if gemini_key:
                try:
                    from google import genai
                    client = genai.Client(api_key=gemini_key)
                    top_ev = season_data.groupby('Batter')['ExitSpeed'].max().nlargest(5).to_dict()
                    top_velo = season_data.groupby('Pitcher')['RelSpeed'].max().nlargest(5).to_dict()
                    system_ctx = f"You are the Marshalls College Baseball League expert scouting AI. Answer concisely (2-4 sentences max). League Context: Top Exit Velo: {top_ev}. Top Pitch Velo: {top_velo}."
                    res = client.models.generate_content(
                        model="gemini-2.5-flash",
                        contents=f"{system_ctx}\n\nQuestion: {user_prompt}"
                    )
                    response_text = res.text
                except Exception:
                    response_text = None

            if not response_text:
                all_b_names = season_data['Batter'].dropna().unique()
                all_p_names = season_data['Pitcher'].dropna().unique()
                matched_batter = next((b for b in all_b_names if any(part.lower() in q for part in b.replace(",", "").split())), None)
                matched_pitcher = next((p for p in all_p_names if any(part.lower() in q for part in p.replace(",", "").split())), None)
                
                if matched_batter:
                    b_sub = season_data[season_data['Batter'] == matched_batter]
                    bip = b_sub[b_sub['ExitSpeed'].notna() & (b_sub['ExitSpeed'] >= 40)]
                    max_ev = bip['ExitSpeed'].max() if not bip.empty else 0
                    avg_ev = bip['ExitSpeed'].mean() if not bip.empty else 0
                    hh_cnt = len(bip[bip['ExitSpeed'] >= 90.0])
                    tot = len(bip)
                    hh_pct = (hh_cnt / tot * 100) if tot > 0 else 0
                    response_text = (
                        f"📊 **Scouting Report for {matched_batter} (Hitter):**\n\n"
                        f"* **Peak Power:** Max Exit Velo of **{max_ev:.1f} mph** with an average EV of **{avg_ev:.1f} mph**.\n"
                        f"* **Hard-Hit Rate (90+ mph):** **{hh_pct:.1f}%** ({hh_cnt} hard-hit balls in {tot} balls in play).\n"
                        f"* **Plate Appearances Tracked:** {len(b_sub)} pitches faced across the active date range."
                    )
                elif matched_pitcher:
                    p_sub = season_data[season_data['Pitcher'] == matched_pitcher]
                    fb = p_sub[p_sub['TaggedPitchType'] == 'Fastball']
                    max_v = fb['RelSpeed'].max() if not fb.empty else p_sub['RelSpeed'].max()
                    avg_v = fb['RelSpeed'].mean() if not fb.empty else p_sub['RelSpeed'].mean()
                    tot_p = len(p_sub)
                    strikes = len(p_sub[p_sub['PitchCall'].astype(str).str.contains("Strike|Foul|InPlay", case=False, na=False)])
                    k_pct = (strikes / tot_p * 100) if tot_p > 0 else 0
                    mix = p_sub['TaggedPitchType'].value_counts(normalize=True).head(3).multiply(100).round(0).to_dict()
                    mix_str = ", ".join([f"{k} ({v:.0f}%)" for k, v in mix.items()])
                    response_text = (
                        f"🎯 **Scouting Report for {matched_pitcher} (Pitcher):**\n\n"
                        f"* **Fastball Velocity:** Tops at **{max_v:.1f} mph**, sitting **{avg_v:.1f} mph**.\n"
                        f"* **Strike Throwing:** Pounding the zone at **{k_pct:.1f}% strikes** over {tot_p} pitches.\n"
                        f"* **Primary Repertoire:** {mix_str}."
                    )
                elif "velo" in q or "fastest" in q or "hardest" in q:
                    top_arms = season_data.groupby('Pitcher')['RelSpeed'].max().nlargest(3).round(1)
                    top_bats = season_data.groupby('Batter')['ExitSpeed'].max().nlargest(3).round(1)
                    response_text = (
                        f"🔥 **League Velocity Leaders (Selected Dates):**\n\n"
                        f"* **Pitchers (Peak FB):** 1. {top_arms.index[0]} ({top_arms.iloc[0]} mph) | 2. {top_arms.index[1]} ({top_arms.iloc[1]} mph) | 3. {top_arms.index[2]} ({top_arms.iloc[2]} mph)\n"
                        f"* **Hitters (Max EV):** 1. {top_bats.index[0]} ({top_bats.iloc[0]} mph) | 2. {top_bats.index[1]} ({top_bats.iloc[1]} mph) | 3. {top_bats.index[2]} ({top_bats.iloc[2]} mph)"
                    )
                else:
                    response_text = "I can inspect any player in the league! Try typing a player's name (e.g. *'How is Ashton Roache performing?'* or *'What is Zayne Hookala's pitch mix?'*)."

            st.markdown(response_text)
            st.session_state["chat_history"].append({"role": "assistant", "content": response_text})

st.divider()

# =====================================================================
# VIEW 1: LEAGUE LEADERBOARD HUB
# =====================================================================
if st.session_state["nav_view"] == "🏆 Leaderboard Hub":
    st.subheader(f"🏆 Marshalls League Official Leaderboard ({selected_year})")
    st.caption("Tap any player card below to open their professional scouting profile.")

    lb_tab_hit, lb_tab_pitch = st.tabs(["💥 Hitting Leaderboards", "🎯 Pitching Leaderboards"])

    with lb_tab_hit:
        batted_all = season_data[season_data['ExitSpeed'].notna() & (season_data['ExitSpeed'] >= 40) & (season_data['Batter'].notna())]
        hitter_agg = batted_all.groupby('Batter').agg(
            BIP=('ExitSpeed', 'count'), Max_EV=('ExitSpeed', 'max'), Avg_EV=('ExitSpeed', 'mean'),
            Hard_Hits=('ExitSpeed', lambda x: (x >= 90.0).sum()),
            Sweet_Spot_Hits=('Angle', lambda x: ((x >= 8.0) & (x <= 32.0)).sum())
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
            FB_Pitches=('RelSpeed', 'count'), Max_FB=('RelSpeed', 'max'), Avg_FB=('RelSpeed', 'mean'),
            Avg_IVB=('InducedVertBreak', 'mean'), Avg_Spin=('SpinRate', 'mean'), Max_Spin=('SpinRate', 'max')
        ).reset_index()

        p_fb_agg['Max_FB'] = p_fb_agg['Max_FB'].round(1)
        p_fb_agg['Avg_FB'] = p_fb_agg['Avg_FB'].round(1)
        p_fb_agg['Avg_IVB'] = p_fb_agg['Avg_IVB'].round(1)
        p_fb_agg['Avg_Spin'] = p_fb_agg['Avg_Spin'].round(0)
        p_fb_agg['Max_Spin'] = p_fb_agg['Max_Spin'].round(0)

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
# VIEW 2: UNIFIED TWO-WAY PLAYER PROFILE (HITTING & PITCHING)
# =====================================================================
elif st.session_state["nav_view"] == "📊 Unified Player Profile":
    if st.button("⬅️ Return to Leaderboard Hub"):
        st.session_state["nav_view"] = "🏆 Leaderboard Hub"
        st.rerun()

    all_p_names = sorted(list(set(season_raw['Batter'].dropna().unique().tolist() + season_raw['Pitcher'].dropna().unique().tolist())))
    default_idx = 0
    if st.session_state.get("selected_player") in all_p_names:
        default_idx = all_p_names.index(st.session_state["selected_player"])

    col_sel, col_gm = st.columns([1.5, 1.5])
    target_player = col_sel.selectbox("Select Player Profile", options=all_p_names, index=default_idx)
    st.session_state["selected_player"] = target_player

    player_all_games = sorted(list(set(
        season_raw[season_raw['Batter'] == target_player]['Game_Source'].dropna().unique().tolist() +
        season_raw[season_raw['Pitcher'] == target_player]['Game_Source'].dropna().unique().tolist()
    )))
    selected_game = col_gm.selectbox("Game Scope Filter", options=["All Games (Cumulative)"] + player_all_games, index=0)

    h_filtered = season_raw[season_raw['Batter'] == target_player]
    p_filtered = season_raw[season_raw['Pitcher'] == target_player]
    if selected_game != "All Games (Cumulative)":
        h_filtered = h_filtered[h_filtered['Game_Source'] == selected_game]
        p_filtered = p_filtered[p_filtered['Game_Source'] == selected_game]

    st.subheader(f"Player Scouting Dossier: **{target_player}**")
    st.caption(f"Scope: {selected_game} | Season: {selected_year}")

    has_hitting = not h_filtered.empty and len(h_filtered[h_filtered['ExitSpeed'].notna()]) > 0
    has_pitching = not p_filtered.empty and len(p_filtered) > 0

    if has_hitting:
        st.markdown("### 💥 **Hitting Performance & Tendencies**")
        bip = h_filtered[h_filtered['ExitSpeed'].notna() & (h_filtered['ExitSpeed'] >= 40)]
        max_ev = bip['ExitSpeed'].max() if not bip.empty else 0
        avg_ev = bip['ExitSpeed'].mean() if not bip.empty else 0
        hard_hits = len(bip[bip['ExitSpeed'] >= 90.0])
        tot_bip = len(bip)
        hh_rate = (hard_hits / tot_bip * 100) if tot_bip > 0 else 0

        hk1, hk2, hk3, hk4 = st.columns(4)
        hk1.metric("Max Exit Velo", f"{max_ev:.1f} mph")
        hk2.metric("Avg Exit Velo", f"{avg_ev:.1f} mph")
        hk3.metric("Hard-Hit Rate (90+)", f"{hard_hits}/{tot_bip} ({hh_rate:.1f}%)")
        hk4.metric("Total BIP", f"{tot_bip}")

        st.markdown("#### **Behavioral Tendencies & Approach Diagnostics**")
        t_col1, t_col2 = st.columns(2)
        with t_col1:
            st.info(f"**Success Indicators:** Hard contact threshold met on {hard_hits} batted balls with peak velocity at {max_ev:.1f} mph. Maintaining optimal barrel path into the zone.")
        with t_col2:
            st.warning("**Failure Indicators / Vulnerabilities:** Whiff frequency elevated against breaking and offspeed offerings away. Recommend aggressive early-count strike selection.")

        st.markdown("#### **Strike Zone Heat Maps (Hard Contact vs. Base Hits)**")
        hm_c1, hm_c2 = st.columns(2)
        with hm_c1:
            st.markdown("*Hard Contact Zone (EV ≥ 90 mph)*")
            st.plotly_chart(render_strike_zone_figure(bip[bip['ExitSpeed'] >= 90.0], "Hard Contact"), use_container_width=True, config={'staticPlot': True})
        with hm_c2:
            st.markdown("*Base Hits Zone*")
            hits_df = bip[bip['PlayResult'].astype(str).str.contains("Single|Double|Triple|HomeRun", case=False, na=False)]
            st.plotly_chart(render_strike_zone_figure(hits_df, "Base Hits"), use_container_width=True, config={'staticPlot': True})

        st.markdown("#### **Field Spray Chart**")
        st.plotly_chart(render_field_spray_chart(bip), use_container_width=True, config={'staticPlot': True})

        st.markdown("#### **Benchmark Comparison Matrix (D1 & MLB Standards)**")
        bench_df = pd.DataFrame({
            "Metric": ["Max Exit Velo", "Average Exit Velo", "Hard-Hit % (90+ mph)", "Sweet-Spot %"],
            "Player Value": [f"{max_ev:.1f} mph", f"{avg_ev:.1f} mph", f"{hh_rate:.1f}%", f"{len(bip[(bip['Angle'] >= 8) & (bip['Angle'] <= 32)])/max(tot_bip, 1)*100:.1f}%"],
            "NCAA Division 1 Benchmark": ["104.5 mph", "89.2 mph", "42.0%", "36.5%"],
            "MLB Standard": ["112.0 mph", "91.5 mph", "48.5%", "38.0%"],
            "Development Action & Instruction": [
                "Maintain bat speed; focus on rotational torque." if max_ev < 100 else "Elite raw power output.",
                "Elevate contact consistency by shortening swing path.",
                "Increase barrel awareness against velocity in the zone." if hh_rate < 40 else "Outstanding hard contact rate.",
                "Tune attack angle to stay in the 8°-32° sweet spot."
            ]
        })
        st.dataframe(bench_df, use_container_width=True, hide_index=True)

    if has_pitching:
        if has_hitting:
            st.divider()
        st.markdown("### 🛡️ **Pitching Arsenal & Spin Rate Profile**")
        total_p = len(p_filtered)
        fb_p = p_filtered[p_filtered['TaggedPitchType'] == 'Fastball']
        max_v = fb_p['RelSpeed'].max() if not fb_p.empty else p_filtered['RelSpeed'].max()
        avg_v = fb_p['RelSpeed'].mean() if not fb_p.empty else p_filtered['RelSpeed'].mean()
        strikes = len(p_filtered[p_filtered['PitchCall'].astype(str).str.contains("Strike|Foul|InPlay", case=False, na=False)])
        strike_pct = (strikes / total_p * 100) if total_p > 0 else 0

        pk1, pk2, pk3, pk4 = st.columns(4)
        pk1.metric("Peak FB Velo", f"{max_v:.1f} mph")
        pk2.metric("Average FB Velo", f"{avg_v:.1f} mph")
        pk3.metric("Total Pitches Tracked", f"{total_p}")
        pk4.metric("Strike %", f"{strike_pct:.1f}%")

        st.markdown("#### **Repertoire Spin Rate Analysis**")
        if 'SpinRate' in p_filtered.columns and 'TaggedPitchType' in p_filtered.columns:
            spin_agg = p_filtered.groupby('TaggedPitchType').agg(
                Count=('RelSpeed', 'count'),
                Avg_Velo=('RelSpeed', 'mean'),
                Avg_Spin=('SpinRate', 'mean'),
                Max_Spin=('SpinRate', 'max'),
                Avg_IVB=('InducedVertBreak', 'mean'),
                Avg_HB=('HorzBreak', 'mean')
            ).reset_index()
            spin_agg['Avg_Velo'] = spin_agg['Avg_Velo'].round(1)
            spin_agg['Avg_Spin'] = spin_agg['Avg_Spin'].round(0)
            spin_agg['Max_Spin'] = spin_agg['Max_Spin'].round(0)
            spin_agg['Avg_IVB'] = spin_agg['Avg_IVB'].round(1)
            spin_agg['Avg_HB'] = spin_agg['Avg_HB'].round(1)
            st.dataframe(spin_agg.rename(columns={'TaggedPitchType': 'Pitch Type', 'Count': 'Pitches', 'Avg_Velo': 'Velo (mph)', 'Avg_Spin': 'Avg Spin (rpm)', 'Max_Spin': 'Max Spin (rpm)', 'Avg_IVB': 'IVB (in)', 'Avg_HB': 'HB (in)'}), use_container_width=True, hide_index=True)

        p_col1, p_col2 = st.columns(2)
        with p_col1:
            st.markdown("#### **Pitch Movement Profile (Static)**")
            fig_mov = px.scatter(
                p_filtered, x="HorzBreak", y="InducedVertBreak", color="TaggedPitchType",
                labels={"HorzBreak": "HB (in)", "InducedVertBreak": "IVB (in)"}, height=320
            )
            fig_mov.update_xaxes(range=[-25, 25], gridcolor="rgba(0,0,0,0.06)")
            fig_mov.update_yaxes(range=[-25, 25], gridcolor="rgba(0,0,0,0.06)")
            fig_mov.add_hline(y=0, line_dash="dash", line_color="#888888")
            fig_mov.add_vline(x=0, line_dash="dash", line_color="#888888")
            fig_mov.update_layout(plot_bgcolor="rgba(245, 247, 250, 0.8)")
            st.plotly_chart(fig_mov, use_container_width=True, config={'staticPlot': True})

        with p_col2:
            st.markdown("#### **Catcher's Location Zone (Static)**")
            st.plotly_chart(render_strike_zone_figure(p_filtered, "Pitcher Location"), use_container_width=True, config={'staticPlot': True})

        st.markdown("#### **Pitcher Benchmark & Spin Rate Comparison Matrix**")
        fb_spin_avg = fb_p['SpinRate'].mean() if not fb_p.empty else 2200
        pitch_bench_df = pd.DataFrame({
            "Metric": ["Fastball Velocity", "Fastball Spin Rate", "Strike Throwing %"],
            "Player Value": [f"{avg_v:.1f} mph", f"{fb_spin_avg:.0f} rpm", f"{strike_pct:.1f}%"],
            "NCAA Division 1 Benchmark": ["91.8 mph", "2320 rpm", "63.5%"],
            "MLB Standard": ["94.2 mph", "2450 rpm", "65.0%"],
            "Development Action & Instruction": [
                "Increase lower half drive and arm speed." if avg_v < 91 else "Elite collegiate velocity.",
                "Optimize spin efficiency and axis extension." if fb_spin_avg < 2250 else "Superior fastball spin characteristics.",
                "Pound the zone early; reduce 3-ball counts."
            ]
        })
        st.dataframe(pitch_bench_df, use_container_width=True, hide_index=True)

# =====================================================================
# VIEW 3: TEAM GAME SUMMARY & MATCH CENTER
# =====================================================================
else:
    all_game_list = sorted([g for g in season_data['Game_Source'].dropna().unique()])
    if not all_game_list:
        st.warning("No games found within the selected date range.")
        st.stop()

    selected_game = st.selectbox("Select Game File", options=all_game_list)
    game_df = season_data[season_data['Game_Source'] == selected_game].copy()

    st.subheader("📋 Official Game Summary & Matchup Intelligence")
    st.caption(f"Game: {selected_game} | Season: {selected_year}")

    clean_name = selected_game.replace(".csv", "")
    if " x " in clean_name:
        parts = clean_name.split(" x ")
        t1_raw = parts[0].split("_")[-1] if "_" in parts[0] else parts[0]
        team_a = re.sub(r"^\d{1,2}-\d{1,2}(-\d{2,4})?\s*", "", t1_raw)
        team_a = re.sub(r"\s*202\d", "", team_a).strip()
        team_b = re.sub(r"\s*202\d", "", parts[1]).strip()
    else:
        team_a, team_b = "Anchors", "Opponent"

    if 'Top/Bottom' in game_df.columns and game_df['Top/Bottom'].dropna().nunique() >= 2:
        t1_bat = game_df[game_df['Top/Bottom'].astype(str).str.lower().str.startswith('top')]
        t2_bat = game_df[game_df['Top/Bottom'].astype(str).str.lower().str.startswith('bot')]
    else:
        mid = len(game_df) // 2
        t1_bat = game_df.iloc[:mid]
        t2_bat = game_df.iloc[mid:]

    t1_pit = t2_bat
    t2_pit = t1_bat

    tab_t1, tab_t2 = st.tabs([f"🛡️ {team_a} Full Game Suite", f"⚔️ {team_b} Full Game Suite"])

    def render_team_game_suite(t_bat, t_pit, current_team_name, opponent_team_name):
        st.markdown(f"### 🎯 **{current_team_name} Hitter Leaderboard (In-Game)**")
        if 'Batter' in t_bat.columns and not t_bat['Batter'].dropna().empty:
            h_lb = t_bat.groupby('Batter').agg(
                Pitches=('PitchCall', 'count'),
                BIP=('ExitSpeed', lambda x: (x >= 40).sum()),
                Max_EV=('ExitSpeed', 'max'),
                Avg_EV=('ExitSpeed', 'mean'),
                Hard_Hits=('ExitSpeed', lambda x: (x >= 90.0).sum()),
                Sweet_Spot=('Angle', lambda x: ((x >= 8.0) & (x <= 32.0)).sum())
            ).reset_index()
            h_lb['Max_EV'] = h_lb['Max_EV'].round(1)
            h_lb['Avg_EV'] = h_lb['Avg_EV'].round(1)
            st.dataframe(h_lb.sort_values(by='Max_EV', ascending=False), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown(f"### 📊 **{current_team_name} Pitching Staff & Sequencing Intelligence**")
        display_shared_sequencing_legend()

        t_pit_enr = enrich_pitch_sequencing_data(t_pit)
        seq_c1, seq_c2, seq_c3 = st.columns(3)
        with seq_c1:
            st.markdown("##### **Pitch Sequencing**")
            st.plotly_chart(render_100pct_stacked_bar(t_pit_enr, 'PitchSeqBucket', ['1P', 'ABR', 'AFB', 'AU', 'AOFF'], False), use_container_width=True, config={'staticPlot': True})
        with seq_c2:
            st.markdown("##### **Count Sequencing**")
            st.plotly_chart(render_100pct_stacked_bar(t_pit_enr, 'CountSeqBucket', ['1P', 'Ahead', 'Behind', 'Even', 'Full'], False), use_container_width=True, config={'staticPlot': True})
        with seq_c3:
            st.markdown("##### **Result Sequencing**")
            st.plotly_chart(render_100pct_stacked_bar(t_pit_enr[t_pit_enr['ResultSeqBucket'].notna()], 'ResultSeqBucket', ['AB', 'ACS', 'AW', 'AF'], False), use_container_width=True, config={'staticPlot': True})

    with tab_t1:
        render_team_game_suite(t1_bat, t1_pit, team_a, team_b)

    with tab_t2:
        render_team_game_suite(t2_bat, t2_pit, team_b, team_a)
