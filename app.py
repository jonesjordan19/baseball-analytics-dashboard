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
                
                # Sanitize outliers
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

def render_strike_zone_figure(df_pitches):
    """Draws a professional catcher's view strike zone with 9 inner quadrants and home plate."""
    fig = go.Figure()

    # 1. Non-contact pitches
    non_contact = df_pitches[df_pitches['ExitSpeed'].isna() | (df_pitches['ExitSpeed'] < 40)]
    for ptype, group in non_contact.groupby('TaggedPitchType'):
        hover_info = group.apply(lambda r: f"{ptype} | {r.get('RelSpeed', 0):.1f} mph<br>Count: {r.get('Balls', 0)}-{r.get('Strikes', 0)}<br>Call: {r.get('PitchCall', '')}", axis=1)
        fig.add_trace(go.Scatter(
            x=group['PlateLocSide'], y=group['PlateLocHeight'],
            mode='markers', name=ptype,
            hovertext=hover_info,
            hoverinfo="text",
            marker=dict(size=12, opacity=0.85)
        ))

    # 2. Contact / Balls in Play (Gold Rings matching WIN Reality)
    contact_p = df_pitches[df_pitches['ExitSpeed'].notna() & (df_pitches['ExitSpeed'] >= 40)]
    if not contact_p.empty:
        hover_contact = contact_p.apply(lambda r: f"CONTACT: {r.get('TaggedPitchType', '')} | {r.get('RelSpeed', 0):.1f} mph<br>EV: {r.get('ExitSpeed', 0):.1f} mph | LA: {r.get('Angle', 0):.0f}°<br>Result: {r.get('PlayResult', '')}", axis=1)
        fig.add_trace(go.Scatter(
            x=contact_p['PlateLocSide'], y=contact_p['PlateLocHeight'],
            mode='markers', name="In Play / Contact",
            hovertext=hover_contact,
            hoverinfo="text",
            marker=dict(size=18, color='rgba(0,0,0,0)',
                        line=dict(color='#FFD700', width=3.5))
        ))

    # 3. Outer Strike Zone Box (-0.83 to +0.83 ft wide, 1.5 to 3.5 ft high)
    fig.add_shape(type="rect", x0=-0.83, y0=1.5, x1=0.83, y1=3.5,
                  line=dict(color="#FFFFFF", width=3))

    # 4. Inner 9-Quadrant Grid Lines
    x_third = 1.66 / 3.0
    y_third = 2.0 / 3.0
    # Vertical grid lines
    fig.add_shape(type="line", x0=-0.83 + x_third, y0=1.5, x1=-0.83 + x_third, y1=3.5,
                  line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=0.83 - x_third, y0=1.5, x1=0.83 - x_third, y1=3.5,
                  line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"))
    # Horizontal grid lines
    fig.add_shape(type="line", x0=-0.83, y0=1.5 + y_third, x1=0.83, y1=1.5 + y_third,
                  line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"))
    fig.add_shape(type="line", x0=-0.83, y0=3.5 - y_third, x1=0.83, y1=3.5 - y_third,
                  line=dict(color="rgba(255,255,255,0.25)", width=1.5, dash="dot"))

    # 5. Home Plate Pentagon at Bottom
    fig.add_trace(go.Scatter(
        x=[-0.708, 0.708, 0.708, 0.0, -0.708, -0.708],
        y=[0.6, 0.6, 0.45, 0.25, 0.45, 0.6],
        fill="toself",
        fillcolor="rgba(200, 200, 200, 0.3)",
        line=dict(color="rgba(255, 255, 255, 0.7)", width=2),
        mode="lines",
        showlegend=False,
        hoverinfo="skip"
    ))

    fig.update_xaxes(range=[-2.2, 2.2], title="Horizontal Plate Location (ft)", zeroline=False, gridcolor="rgba(255,255,255,0.08)")
    fig.update_yaxes(range=[0.0, 4.5], title="Height from Ground (ft)", zeroline=False, gridcolor="rgba(255,255,255,0.08)")
    fig.update_layout(template="plotly_dark", height=450, margin=dict(l=20, r=20, t=30, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

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

report_scope = st.sidebar.radio("View Report Level", [
    "🔥 Individual Hitter Card",
    "🛡️ Individual Pitcher Card",
    "🏆 Team Game Summary"
])

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
# 1. INDIVIDUAL HITTER REPORT CARD
# =====================================================================
if report_scope == "🔥 Individual Hitter Card":
    batters = sorted([b for b in active_data['Batter'].dropna().unique() if str(b).strip()])
    if not batters:
        st.warning("No hitter data tracked for this selection.")
        st.stop()

    selected_batter = st.sidebar.selectbox("Select Batter", options=batters)
    b_data = active_data[active_data['Batter'] == selected_batter].copy()

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
    st.caption(f"Game: {selected_game} | Season: {selected_year}")

    # AI TAKEAWAYS
    t1, t2 = st.columns(2)
    with t1:
        if max_ev >= 95:
            st.success(f"**Barrel was loud:** 100+ exit velo recorded ({max_ev:.1f} mph). Power on line drives.")
        elif avg_ev >= 88:
            st.success(f"**Consistent contact:** Solid contact quality averaging {avg_ev:.1f} mph off the bat.")
        else:
            st.info("**Working the counts:** Saw pitches and fought into deep counts.")

    with t2:
        if len(ground_balls) > len(fly_balls) and len(in_play) > 0:
            st.warning(f"**Pick it up:** {len(ground_balls)} of {len(in_play)} balls in play stayed on the ground. Match the pitch plane and elevate.")
        elif len(sweet_spot) > 0:
            st.success(f"**Good angles:** {len(sweet_spot)} of {len(in_play)} balls in play were squared in the 8°-32° sweet-spot zone.")
        else:
            st.info("**Aggression on strikes:** Attack early count fastballs in the strike zone.")

    # TOP METRICS
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Hard-Hit Rate (90+)", f"{len(hard_hits)}/{len(in_play)}" if len(in_play) > 0 else "0/0")
    k2.metric("Average Exit Velo", f"{avg_ev:.1f} mph" if avg_ev > 0 else "N/A")
    k3.metric("Max Exit Velo", f"{max_ev:.1f} mph" if max_ev > 0 else "N/A")
    k4.metric("Max Distance", f"{max_dist:.0f} ft" if max_dist > 0 else "N/A")
    k5.metric("Pitches / Swings", f"{total_pitches} / {swings}")

    st.divider()

    # ROW 1: PRO STRIKE ZONE + AT-BAT SEQUENCING TIMELINE
    col_zone, col_seq = st.columns([1.1, 1.3])

    with col_zone:
        st.markdown("#### **Pitches Seen (Catcher's View)**")
        if 'PlateLocSide' in b_data.columns and 'PlateLocHeight' in b_data.columns:
            st.plotly_chart(render_strike_zone_figure(b_data), use_container_width=True)
        else:
            st.info("Plate location coordinates are unavailable.")

    with col_seq:
        st.markdown("#### **At-Bat Pitch Sequencing Timeline**")
        st.caption("Chronological progression of every pitch seen in each plate appearance:")
        
        # Build chronological pitch sequence
        sort_cols = [c for c in ['Inning', 'PAofInning', 'PitchofPA', 'Time'] if c in b_data.columns]
        if sort_cols:
            b_sorted = b_data.sort_values(by=sort_cols).copy()
        else:
            b_sorted = b_data.copy()

        timeline_data = []
        for idx, r in b_sorted.iterrows():
            inn = r.get('Inning', '-')
            p_num = r.get('PitchofPA', '-')
            count_str = f"{int(r.get('Balls', 0))}-{int(r.get('Strikes', 0))}"
            ptype = r.get('TaggedPitchType', 'Unknown')
            velo = f"{r.get('RelSpeed', 0):.1f} mph" if pd.notna(r.get('RelSpeed')) else "-"
            call = r.get('PitchCall', '-')
            
            # Format EV / LA if ball in play
            ev = r.get('ExitSpeed', None)
            la = r.get('Angle', None)
            res = r.get('PlayResult', None)
            
            if pd.notna(ev) and ev >= 40:
                outcome = f"💥 In Play: {ev:.1f} mph, {la:.0f}° ({res if pd.notna(res) else 'Contact'})"
            else:
                outcome = call

            timeline_data.append({
                "Inn": inn,
                "Pitch #": p_num,
                "Count": count_str,
                "Pitch Type": ptype,
                "Velo": velo,
                "Pitch Call / Outcome": outcome
            })
            
        if timeline_data:
            st.dataframe(pd.DataFrame(timeline_data), use_container_width=True, hide_index=True)
        else:
            st.info("No pitch sequence data available.")

    st.divider()

    # ROW 2: COUNT SEQUENCING BREAKDOWN TABLE
    st.markdown("#### **Count Sequencing — What Did The Opposition Throw You?**")
    st.caption("Breakdown of pitch types thrown to this batter based on game situation:")

    def get_count_bucket(r):
        b = int(r.get('Balls', 0))
        s = int(r.get('Strikes', 0))
        if b == 0 and s == 0:
            return "1st Pitch (0-0)"
        elif s == 2:
            return "2 Strikes (0-2, 1-2, 2-2, 3-2)"
        elif b > s:
            return "Hitter's Count (Ahead: 1-0, 2-0, 2-1, 3-0, 3-1)"
        elif s > b:
            return "Pitcher's Count (Behind: 0-1)"
        elif b == s and b > 0:
            return "Even Count (1-1)"
        return "Other"

    b_data['CountState'] = b_data.apply(get_count_bucket, axis=1)
    
    order = ["1st Pitch (0-0)", "Hitter's Count (Ahead: 1-0, 2-0, 2-1, 3-0, 3-1)", "Pitcher's Count (Behind: 0-1)", "Even Count (1-1)", "2 Strikes (0-2, 1-2, 2-2, 3-2)"]
    
    seq_matrix = pd.crosstab(b_data['CountState'], b_data['TaggedPitchType'], normalize='index').multiply(100).round(0)
    count_totals = b_data['CountState'].value_counts()
    seq_matrix['Total Pitches'] = count_totals
    
    # Reindex to logical baseball order
    present_order = [o for o in order if o in seq_matrix.index]
    seq_matrix = seq_matrix.reindex(present_order)
    st.dataframe(seq_matrix.fillna(0).astype(int), use_container_width=True)

# =====================================================================
# 2. INDIVIDUAL PITCHER REPORT CARD
# =====================================================================
elif report_scope == "🛡️ Individual Pitcher Card":
    pitchers = sorted([p for p in active_data['Pitcher'].dropna().unique() if str(p).strip()])
    if not pitchers:
        st.warning("No pitcher data tracked for this selection.")
        st.stop()

    selected_pitcher = st.sidebar.selectbox("Select Pitcher", options=pitchers)
    p_data = active_data[active_data['Pitcher'] == selected_pitcher].copy()

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
        st.markdown("#### **Location & Damage Allowed (Catcher's View)**")
        if 'PlateLocSide' in p_data.columns and 'PlateLocHeight' in p_data.columns:
            st.plotly_chart(render_strike_zone_figure(p_data), use_container_width=True)
        else:
            st.info("Plate location coordinates are unavailable.")

    st.divider()

    st.markdown("#### **Pitcher Count Sequencing Usage**")
    def categorize_pitcher_count(row):
        b = int(row.get('Balls', 0))
        s = int(row.get('Strikes', 0))
        if b == 0 and s == 0: return '1st Pitch (0-0)'
        if s > b: return 'Ahead'
        if b > s: return 'Behind'
        if b == s and b > 0: return 'Even'
        return 'Other'
        
    p_data['CountState'] = p_data.apply(categorize_pitcher_count, axis=1)
    seq = pd.crosstab(p_data['CountState'], p_data['TaggedPitchType'], normalize='index').multiply(100).round(0)
    st.dataframe(seq.astype(int), use_container_width=True)

# =====================================================================
# 3. TEAM GAME SUMMARY
# =====================================================================
else:
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
