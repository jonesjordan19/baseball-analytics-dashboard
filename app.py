import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
                
                # Filter sensor glitches
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

# ----------------- BRANDING & HEADER -----------------
st.title("⚡ Marshalls League Data Engine")
st.markdown("##### **Created by Jordan Jones** | *Official WIN Reality SmartPark Analytics & Scouting Suite*")

with st.spinner("Connecting to Marshalls Cloud Lake..."):
    data = load_marshalls_telemetry(MANIFEST_SHEET_ID)

if data.empty:
    st.warning("Telemetry is loading. Please check permissions on the Google Sheet.")
    st.stop()

# ----------------- SIDEBAR FILTERS -----------------
st.sidebar.header("🎯 Game & Player Controls")

# Report View Level
report_scope = st.sidebar.radio("View Report Level", [
    "🔥 Individual Hitter Card",
    "🛡️ Individual Pitcher Card",
    "🏆 Team Game Summary"
])

# Season & Game Filters
available_years = sorted(data['Season_Year'].dropna().unique())
selected_year = st.sidebar.selectbox("Season Year", options=available_years, index=0)
season_data = data[data['Season_Year'] == selected_year]

games = ["All Games (Season Cumulative)"] + sorted([g for g in season_data['Game_Source'].dropna().unique()])
selected_game = st.sidebar.selectbox("Game Select", options=games)

if selected_game != "All Games (Season Cumulative)":
    active_data = season_data[season_data['Game_Source'] == selected_game]
else:
    active_data = season_data

st.divider()

# =====================================================================
# 1. INDIVIDUAL HITTER REPORT CARD (WIN REALITY REPLICA)
# =====================================================================
if report_scope == "🔥 Individual Hitter Card":
    batters = sorted([b for b in active_data['Batter'].dropna().unique() if str(b).strip()])
    if not batters:
        st.warning("No hitter data tracked for this selection.")
        st.stop()

    selected_batter = st.sidebar.selectbox("Select Batter", options=batters)
    b_data = active_data[active_data['Batter'] == selected_batter]

    # Metrics computation
    total_pitches = len(b_data)
    swings = len(b_data[b_data['PitchCall'].astype(str).str.contains("StrikeSwinging|Foul|InPlay", case=False, na=False)])
    in_play = b_data[b_data['ExitSpeed'].notna() & (b_data['ExitSpeed'] > 40)]
    hard_hits = in_play[in_play['ExitSpeed'] >= 90.0]
    sweet_spot = in_play[(in_play['Angle'] >= 8.0) & (in_play['Angle'] <= 32.0)]
    ground_balls = in_play[in_play['Angle'] < 8.0]
    fly_balls = in_play[in_play['Angle'] > 32.0]
    
    # 2-Strike and Hitter's counts
    hitter_counts = len(b_data[(b_data['Balls'] >= 2) & (b_data['Strikes'] <= 1)])
    two_strike_pitches = len(b_data[b_data['Strikes'] == 2])
    
    avg_ev = in_play['ExitSpeed'].mean() if not in_play.empty else 0
    max_ev = in_play['ExitSpeed'].max() if not in_play.empty else 0
    max_dist = in_play['Distance'].max() if 'Distance' in in_play.columns and in_play['Distance'].notna().any() else 0

    st.subheader(f"Hitter Postgame Report: **{selected_batter}**")
    st.caption(f"Game: {selected_game} | Season: {selected_year}")

    # AI TAKEAWAYS BLOCK
    takeaway_col1, takeaway_col2 = st.columns(2)
    with takeaway_col1:
        if max_ev >= 95:
            st.success(f"**Barrel was loud:** Topped at **{max_ev:.1f} mph** off the bat. Real collegiate juice.")
        elif avg_ev >= 88:
            st.success(f"**Consistent contact:** Maintained an **{avg_ev:.1f} mph** average exit velocity.")
        else:
            st.info("**Working the counts:** Saw pitches and fought into deep counts.")

    with takeaway_col2:
        if len(ground_balls) > len(fly_balls) and len(in_play) > 0:
            st.warning(f"**Elevate:** {len(ground_balls)} of {len(in_play)} balls in play stayed on the ground. Match the pitch plane and lift into the sweet spot.")
        elif len(sweet_spot) > 0:
            st.success(f"**Good angles:** {len(sweet_spot)} of {len(in_play)} balls in play were squared on a line in the 8°-32° sweet-spot zone.")
        else:
            st.info("**Stay aggressive early:** Jump on first-pitch fastballs in the strike zone.")

    # TOP KPI METRIC ROW
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Hard-Hit Rate (90+)", f"{len(hard_hits)}/{len(in_play)}" if len(in_play) > 0 else "0/0")
    k2.metric("Average Exit Velo", f"{avg_ev:.1f} mph" if avg_ev > 0 else "N/A")
    k3.metric("Max Exit Velo", f"{max_ev:.1f} mph" if max_ev > 0 else "N/A")
    k4.metric("Max Distance", f"{max_dist:.0f} ft" if max_dist > 0 else "N/A")
    k5.metric("Pitches / Swings", f"{total_pitches} / {swings}")

    st.divider()

    # TWO COLUMNS: ZONE WITH CONTACT RINGS + AT-BAT LOG
    c_zone, c_log = st.columns([1, 1.4])

    with c_zone:
        st.markdown("#### **Pitches Seen (Catcher's View)**")
        if 'PlateLocSide' in b_data.columns and 'PlateLocHeight' in b_data.columns:
            fig_sz = go.Figure()

            # Plot non-contact pitches
            non_contact = b_data[b_data['ExitSpeed'].isna()]
            for ptype, group in non_contact.groupby('TaggedPitchType'):
                fig_sz.add_trace(go.Scatter(
                    x=group['PlateLocSide'], y=group['PlateLocHeight'],
                    mode='markers', name=ptype,
                    marker=dict(size=11, opacity=0.85)
                ))

            # Plot contact pitches with bright Gold Rings (just like WIN SmartPark)
            contact_p = b_data[b_data['ExitSpeed'].notna()]
            if not contact_p.empty:
                fig_sz.add_trace(go.Scatter(
                    x=contact_p['PlateLocSide'], y=contact_p['PlateLocHeight'],
                    mode='markers', name="In Play / Contact",
                    marker=dict(size=17, color='rgba(0,0,0,0)',
                                line=dict(color='#FFD700', width=3.5)),
                    hovertext=contact_p['ExitSpeed'].apply(lambda x: f"EV: {x:.1f} mph")
                ))

            # Draw MLB Standard Strike Zone outline
            fig_sz.add_shape(type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
                             line=dict(color="white", width=2.5))
            fig_sz.update_xaxes(range=[-2.2, 2.2], title="Horizontal Plate Location (ft)")
            fig_sz.update_yaxes(range=[0.5, 4.5], title="Height from Ground (ft)")
            fig_sz.update_layout(template="plotly_dark", height=420, margin=dict(l=20, r=20, t=30, b=20))
            st.plotly_chart(fig_sz, use_container_width=True)

    with c_log:
        st.markdown("#### **At-Bat Action Log**")
        log_cols = ['Inning', 'Balls', 'Strikes', 'PitchCall', 'ExitSpeed', 'Angle', 'HitType', 'PlayResult']
        available_log = [c for c in log_cols if c in b_data.columns]
        if not in_play.empty:
            display_log = in_play[available_log].rename(columns={
                'ExitSpeed': 'EV (mph)',
                'Angle': 'LA (°)',
                'PlayResult': 'Outcome'
            })
            st.dataframe(display_log.round(1), use_container_width=True, hide_index=True)
        else:
            st.info("No balls put into play for this hitter in the selected frame.")

# =====================================================================
# 2. INDIVIDUAL PITCHER REPORT CARD (WIN REALITY REPLICA)
# =====================================================================
elif report_scope == "🛡️ Individual Pitcher Card":
    pitchers = sorted([p for p in active_data['Pitcher'].dropna().unique() if str(p).strip()])
    if not pitchers:
        st.warning("No pitcher data tracked for this selection.")
        st.stop()

    selected_pitcher = st.sidebar.selectbox("Select Pitcher", options=pitchers)
    p_data = active_data[active_data['Pitcher'] == selected_pitcher]

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
    st.caption(f"Game: {selected_game} | Season: {selected_year}")

    # AI COACHING TAKEAWAYS
    col_t1, col_t2 = st.columns(2)
    with col_t1:
        if fp_strike_pct >= 65:
            st.success(f"**Pounded the zone early:** {fp_strike_pct:.0f}% first-pitch strikes. Set the tone and dictated counts.")
        else:
            st.warning(f"**Mix pitch one:** First-pitch strike rate was {fp_strike_pct:.0f}%. Get ahead early to open up your secondaries.")
    with col_t2:
        if strike_pct >= 62:
            st.success(f"**In the zone all day:** {strike_pct:.0f}% total strikes. Filled up the zone and challenged bats.")
        else:
            st.info(f"**Count control:** Attack the 1-1 and 2-2 even counts to stay out of hitter counts.")

    # TOP KPI ROW
    pk1, pk2, pk3, pk4, pk5 = st.columns(5)
    pk1.metric("Total Pitches", f"{total_p}")
    pk2.metric("Strike %", f"{strike_pct:.0f}%")
    pk3.metric("Avg FB Velo", f"{avg_fb:.1f} mph" if avg_fb > 0 else "N/A")
    pk4.metric("Max FB Velo", f"{max_fb:.1f} mph" if max_fb > 0 else "N/A")
    pk5.metric("First-Pitch Strike %", f"{fp_strike_pct:.0f}%")

    st.divider()

    # PITCH MOVEMENT (IVB vs HB) + COUNT SEQUENCING
    p_col1, p_col2 = st.columns([1.2, 1])

    with p_col1:
        st.markdown("#### **Pitch Movement (Pitcher's View)**")
        fig_mov = px.scatter(
            p_data,
            x="HorzBreak",
            y="InducedVertBreak",
            color="TaggedPitchType",
            hover_data=["RelSpeed", "SpinRate"],
            labels={"HorzBreak": "Horizontal Break (HB) [in]", "InducedVertBreak": "Induced Vertical Break (IVB) [in]"},
            template="plotly_dark",
            height=400
        )
        fig_mov.update_xaxes(range=[-25, 25])
        fig_mov.update_yaxes(range=[-25, 25])
        fig_mov.add_hline(y=0, line_dash="dash", line_color="#888888")
        fig_mov.add_vline(x=0, line_dash="dash", line_color="#888888")
        st.plotly_chart(fig_mov, use_container_width=True)

    with p_col2:
        st.markdown("#### **Pitch Metrics by Type**")
        p_table = p_data.groupby('TaggedPitchType').agg({
            'RelSpeed': ['count', 'mean', 'max'],
            'SpinRate': 'mean',
            'InducedVertBreak': 'mean',
            'HorzBreak': 'mean'
        }).round(1)
        p_table.columns = ['Count', 'Avg Velo', 'Max Velo', 'Spin', 'IVB', 'HB']
        st.dataframe(p_table.reset_index(), use_container_width=True, hide_index=True)

        st.markdown("#### **Count Sequencing Usage**")
        if 'Balls' in p_data.columns and 'Strikes' in p_data.columns:
            def categorize_count(row):
                if row['Balls'] == 0 and row['Strikes'] == 0: return '1st Pitch'
                if row['Strikes'] > row['Balls']: return 'Ahead'
                if row['Balls'] > row['Strikes']: return 'Behind'
                if row['Balls'] == row['Strikes'] and row['Balls'] > 0: return 'Even'
                return 'Other'
            p_data['CountState'] = p_data.apply(categorize_count, axis=1)
            seq = pd.crosstab(p_data['CountState'], p_data['TaggedPitchType'], normalize='index').multiply(100).round(0)
            st.dataframe(seq, use_container_width=True)

# =====================================================================
# 3. TEAM GAME SUMMARY (POST-GAME TARGET BARS)
# =====================================================================
else:
    st.subheader(f"Team Postgame Benchmark Report")
    st.caption(f"Game: {selected_game} | Season: {selected_year}")

    batted_team = active_data[active_data['ExitSpeed'].notna() & (active_data['ExitSpeed'] > 40)]
    hard_hits_team = len(batted_team[batted_team['ExitSpeed'] >= 90.0])
    sweet_spot_team = len(batted_team[(batted_team['Angle'] >= 8.0) & (batted_team['Angle'] <= 32.0)])
    
    total_swings = len(active_data[active_data['PitchCall'].astype(str).str.contains("StrikeSwinging|Foul|InPlay", case=False, na=False)])
    two_strike_team = len(active_data[active_data['Strikes'] == 2])
    
    hh_rate = (hard_hits_team / len(batted_team) * 100) if len(batted_team) > 0 else 0
    sw_rate = (sweet_spot_team / len(batted_team) * 100) if len(batted_team) > 0 else 0

    st.markdown("#### **Team Performance vs. College Targets**")
    
    # Target Progress Bars styled after WIN SmartPark benchmarks
    t1, t2 = st.columns(2)
    with t1:
        st.write(f"**Hard-Hit Rate (90+ MPH): {hh_rate:.0f}%** *(Target: 40%+)*")
        st.progress(min(int(hh_rate) / 100, 1.0))
        
        st.write(f"**Launch Angle Sweet-Spot % (8°-32°): {sw_rate:.0f}%** *(Target: 35%+)*")
        st.progress(min(int(sw_rate) / 100, 1.0))

    with t2:
        gb_count = len(batted_team[batted_team['Angle'] < 8])
        ld_count = len(batted_team[(batted_team['Angle'] >= 8) & (batted_team['Angle'] <= 32)])
        fb_count = len(batted_team[batted_team['Angle'] > 32])
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
