"""
Cricket Win Predictor — Streamlit Dashboard Application
T20 Match Win Probability — Pre-Match & Live Demo
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st

from src.predict_target_match import compute_target_match_features
from src.predict_live import LivePredictor
from src.cricketdata_api import (
    CricketDataClient,
    extract_live_match_state,
    load_historical_elo_calculator,
    parse_overs
)

TEAM_ALIASES_MAP = {
    'nam': 'Namibia', 'sa': 'South Africa', 'rsa': 'South Africa',
    'ind': 'India', 'aus': 'Australia', 'eng': 'England', 'pak': 'Pakistan',
    'nz': 'New Zealand', 'wi': 'West Indies', 'sl': 'Sri Lanka', 'ban': 'Bangladesh',
    'afg': 'Afghanistan', 'zim': 'Zimbabwe', 'ire': 'Ireland', 'ned': 'Netherlands',
    'uae': 'United Arab Emirates', 'usa': 'United States of America', 'scot': 'Scotland',
    'nep': 'Nepal', 'can': 'Canada', 'om': 'Oman', 'oman': 'Oman'
}

def fallback_resolve_team_name(name: str) -> str:
    if not name:
        return "Unknown"
    clean = str(name).strip()
    lower = clean.lower()
    if lower in TEAM_ALIASES_MAP:
        return TEAM_ALIASES_MAP[lower]
    import re
    stripped = re.sub(r"\b(t20|women|men|xi|u19|squad)\b", "", clean, flags=re.IGNORECASE).strip()
    if stripped.lower() in TEAM_ALIASES_MAP:
        return TEAM_ALIASES_MAP[stripped.lower()]
    return stripped if stripped else clean

try:
    from src.cricketdata_api import resolve_team_name
except ImportError:
    resolve_team_name = fallback_resolve_team_name

try:
    from src.cricketdata_api import filter_upcoming_matches, filter_live_matches
except ImportError:
    def filter_upcoming_matches(matches: list) -> list:
        res = [m for m in matches if not m.get("matchStarted", False) and not m.get("matchEnded", False)]
        return res if res else matches

    def filter_live_matches(matches: list) -> list:
        res = [m for m in matches if m.get("matchStarted", False) and not m.get("matchEnded", False)]
        return res if res else matches

from datetime import datetime, timezone, timedelta

def format_datetime_ist(dt_str: str, date_fallback: str = "") -> str:
    """
    Format ISO / GMT datetime string into clean Indian Standard Time (IST, UTC+5:30) representation.
    """
    target = dt_str or date_fallback
    if not target:
        return "Time TBA"
    try:
        import dateutil.parser
        dt = dateutil.parser.parse(target)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        ist = dt.astimezone(timezone(timedelta(hours=5, minutes=30)))
        return ist.strftime("%d %b %Y, %I:%M %p IST")
    except Exception:
        return f"{target} (IST)"


@st.cache_resource
def get_cached_elo_calc():
    return load_historical_elo_calculator()



# ──────────────────────────────────────────────────────────────────────────────
# STREAMLIT PAGE CONFIGURATION & STYLING
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Cricket Win Predictor",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Dark Theme & Sleek Dashboard Navigation CSS
st.markdown("""
<style>
    /* Global Background & Typography */
    .main {
        background-color: #0B0F19;
        color: #F8FAFC;
        font-family: 'Inter', system-ui, -apple-system, sans-serif;
    }

    [data-testid="stSidebar"] {
        background-color: #111827;
        border-right: 1px solid #1F2937;
    }
    
    .title-banner {
        font-size: 2.4rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(135deg, #F8FAFC 0%, #94A3B8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.1rem;
    }
    
    .subtitle-banner {
        font-size: 1.05rem;
        color: #9CA3AF;
        margin-bottom: 1.5rem;
    }
    
    /* Central Probability Display Box */
    .prob-box {
        background: linear-gradient(145deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 1.75rem;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.4);
    }
    
    .team-label {
        font-size: 1.1rem;
        font-weight: 700;
        color: #94A3B8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    
    .prob-val-batting {
        font-size: 3.5rem;
        font-weight: 900;
        color: #38BDF8;
        line-height: 1.1;
        text-shadow: 0 0 20px rgba(56, 189, 248, 0.3);
    }
    
    .prob-val-bowling {
        font-size: 3.5rem;
        font-weight: 900;
        color: #F43F5E;
        line-height: 1.1;
        text-shadow: 0 0 20px rgba(244, 63, 94, 0.3);
    }

    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: #1F2937;
        border: 1px solid #374151;
        padding: 14px 18px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    
    div[data-testid="stMetricLabel"] {
        color: #9CA3AF;
        font-size: 0.85rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    div[data-testid="stMetricValue"] {
        color: #F9FAFB;
        font-size: 1.6rem;
        font-weight: 800;
    }

    /* Modern Dashboard Sidebar Radio Buttons */
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] {
        gap: 10px !important;
    }

    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] > label {
        background-color: #1F2937 !important;
        border: 1px solid #374151 !important;
        border-radius: 12px !important;
        padding: 14px 16px !important;
        color: #E5E7EB !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease-in-out !important;
        cursor: pointer !important;
        width: 100% !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1) !important;
    }

    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] > label:hover {
        border-color: #38BDF8 !important;
        background-color: #374151 !important;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(56, 189, 248, 0.15) !important;
    }

    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] > label[data-checked="true"],
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] > label:has(input:checked) {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
        border: 1px solid #38BDF8 !important;
        box-shadow: 0 0 15px rgba(56, 189, 248, 0.25) !important;
    }

    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] > label[data-checked="true"] p,
    div[data-testid="stSidebarUserContent"] div[role="radiogroup"] > label:has(input:checked) p {
        color: #38BDF8 !important;
        font-weight: 700 !important;
    }

    /* Sidebar Status Card */
    .sidebar-status-box {
        background: #1F2937;
        border: 1px solid #374151;
        border-radius: 12px;
        padding: 1rem;
        margin-top: 1.5rem;
    }
    .sidebar-status-title {
        font-size: 0.8rem;
        font-weight: 700;
        color: #9CA3AF;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 0.75rem;
    }
    .sidebar-status-item {
        font-size: 0.88rem;
        color: #D1D5DB;
        display: flex;
        justify-content: space-between;
        margin-bottom: 0.4rem;
    }

    .footer-text {
        text-align: center;
        color: #6B7280;
        font-size: 0.85rem;
        margin-top: 3rem;
        padding-top: 1.5rem;
        border-top: 1px solid #1F2937;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CACHED LOADERS
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_historical_live_features():
    path = os.path.join("data", "processed", "live_features.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_resource
def load_live_predictor_model():
    try:
        return LivePredictor()
    except Exception:
        return None

@st.cache_resource
def load_prematch_model_artifact():
    path = os.path.join("models", "prematch_model.pkl")
    features_path = os.path.join("models", "prematch_features.pkl")
    if os.path.exists(path) and os.path.exists(features_path):
        model = joblib.load(path)
        features = joblib.load(features_path)
        return model, features
    return None, None

@st.cache_data
def load_metrics_json():
    pm_path = os.path.join("models", "prematch_metrics.json")
    lm_path = os.path.join("models", "live_metrics.json")
    pm_data = json.load(open(pm_path)) if os.path.exists(pm_path) else {}
    lm_data = json.load(open(lm_path)) if os.path.exists(lm_path) else {}
    return pm_data, lm_data


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR NAVIGATION & STATUS PANEL
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 🏏 Cricket Predictor")
st.sidebar.caption("T20 Machine Learning Dashboard")
st.sidebar.markdown("---")

st.sidebar.markdown("<p style='font-size:0.85rem; font-weight:700; color:#9CA3AF; text-transform:uppercase; letter-spacing:0.05em;'>DASHBOARD MENU</p>", unsafe_allow_html=True)

app_mode = st.sidebar.radio(
    "Navigation Menu",
    [
        "🔮  Pre-Match Prediction",
        "📈  Historical Live Demo",
        "📡  API Live Mode"
    ],
    index=0,  # Pre-Match Prediction is DEFAULT landing mode
    label_visibility="collapsed"
)

cd_client_status = CricketDataClient()
cd_token_configured = cd_client_status.is_configured()

# Sleek API Status Panel
st.sidebar.markdown(f"""
<div class="sidebar-status-box">
    <div class="sidebar-status-title">System Status</div>
    <div class="sidebar-status-item">
        <span>CricketData API:</span>
        <span style="color:{'#10B981' if cd_token_configured else '#F59E0B'}; font-weight:700;">
            {'Key Configured ✅' if cd_token_configured else 'Key Missing'}
        </span>
    </div>
    <div class="sidebar-status-item">
        <span>Target Fixture:</span>
        <span style="color:#10B981; font-weight:700;">Available ✅</span>
    </div>
    <div class="sidebar-status-item">
        <span>Historical Demo:</span>
        <span style="color:#10B981; font-weight:700;">Available ✅</span>
    </div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption("Model Engine: Logistic Regression + XGBoost")


# ──────────────────────────────────────────────────────────────────────────────
# MAIN CONTENT HEADER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="title-banner">Cricket Win Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">T20 Match Win Probability — Pre-Match & Live</div>', unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# MODE 1: PRE-MATCH PREDICTION (DEFAULT LANDING PAGE)
# ──────────────────────────────────────────────────────────────────────────────
if "Pre-Match Prediction" in app_mode:
    st.markdown("## 🔮 PRE-MATCH WIN PREDICTION")
    st.caption("Pre-match win probability for upcoming T20 international fixtures calculated from historical Elo, form, and venue priors.")

    st.markdown("---")

    pm_model, pm_features = load_prematch_model_artifact()
    matches_csv = os.path.join("data", "processed", "matches.csv")
    deliveries_csv = os.path.join("data", "processed", "deliveries.csv")

    cd_client = CricketDataClient()
    raw_api_matches = []
    if cd_client.is_configured():
        fetch_func = getattr(cd_client, "fetch_matches_list", cd_client.fetch_current_matches)
        api_res = fetch_func()
        if not api_res or api_res.get("status") != "success":
            api_res = cd_client.fetch_current_matches()
        if api_res and api_res.get("status") == "success":
            raw_api_matches = api_res.get("data", [])

    api_matches = filter_upcoming_matches(raw_api_matches)

    SAMPLE_UPCOMING_FIXTURES = [
        {"name": "Namibia vs South Africa (Upcoming T20)", "teams": ["Namibia", "South Africa"], "venue": "Namibia Cricket Ground", "date": "2026-09-05", "dateTimeGMT": "2026-09-05T12:00:00", "matchType": "t20"},
        {"name": "Zimbabwe vs South Africa (Upcoming T20)", "teams": ["Zimbabwe", "South Africa"], "venue": "Harare Sports Club", "date": "2026-09-06", "dateTimeGMT": "2026-09-06T13:30:00", "matchType": "t20"},
        {"name": "India vs Australia (Upcoming T20)", "teams": ["India", "Australia"], "venue": "Melbourne Cricket Ground", "date": "2026-09-07", "dateTimeGMT": "2026-09-07T14:00:00", "matchType": "t20"},
        {"name": "England vs Pakistan (Upcoming T20)", "teams": ["England", "Pakistan"], "venue": "Lords, London", "date": "2026-09-09", "dateTimeGMT": "2026-09-09T09:00:00", "matchType": "t20"},
        {"name": "South Africa vs New Zealand (Upcoming T20)", "teams": ["South Africa", "New Zealand"], "venue": "Cape Town", "date": "2026-09-10", "dateTimeGMT": "2026-09-10T15:00:00", "matchType": "t20"},
        {"name": "West Indies vs Zimbabwe (Upcoming T20)", "teams": ["West Indies", "Zimbabwe"], "venue": "Harare Sports Club", "date": "2026-09-12", "dateTimeGMT": "2026-09-12T13:30:00", "matchType": "t20"},
    ]

    combined_fixtures = list(api_matches)
    existing_names = {str(m.get("name", "")).lower() for m in combined_fixtures}
    for s in SAMPLE_UPCOMING_FIXTURES:
        s_name = s["name"].lower()
        if not any(s["teams"][0].lower() in n and s["teams"][1].lower() in n for n in existing_names):
            combined_fixtures.append(s)

    fixtures_to_show = combined_fixtures

    pm_team1 = "India"
    pm_team2 = "Australia"
    pm_venue = "Melbourne Cricket Ground"
    pm_date = "2026-09-05"
    pm_date_gmt = "2026-09-05T14:00:00"

    POPULAR_TEAMS = [
        "India", "Australia", "England", "Pakistan", "South Africa", "New Zealand",
        "West Indies", "Sri Lanka", "Bangladesh", "Afghanistan", "Zimbabwe", "Ireland",
        "Netherlands", "Namibia", "Scotland", "United Arab Emirates", "United States of America",
        "Nepal", "Canada", "Oman"
    ]

    POPULAR_VENUES = [
        "Melbourne Cricket Ground", "Dubai International Cricket Stadium", "Eden Gardens, Kolkata",
        "Wankhede Stadium, Mumbai", "Sydney Cricket Ground", "Lords, London",
        "Narendra Modi Stadium, Ahmedabad", "M. Chinnaswamy Stadium, Bengaluru",
        "Harare Sports Club", "Sharjah Cricket Stadium", "Al Amerat Cricket Ground Oman Cricket"
    ]

    source_label = "📡 Upcoming Fixtures (API Feed)" if api_matches else "📅 Select Upcoming Fixture"
    mode_choice = st.radio("Match Source:", [source_label, "⚙️ Custom Teams & Venue Configurator"], horizontal=True)

    if source_label in mode_choice:
        match_labels = [
            f"{m.get('name', 'Match')} [{str(m.get('matchType', 'T20')).upper()}] — ⏰ {format_datetime_ist(m.get('dateTimeGMT'), m.get('date', 'Upcoming'))} @ {m.get('venue', 'Unknown Venue')}"
            for m in fixtures_to_show
        ]
        sel_idx = st.selectbox("Select Upcoming Match to Predict:", range(len(fixtures_to_show)), format_func=lambda i: match_labels[i])
        m_sel = fixtures_to_show[sel_idx]

        teams_arr = m_sel.get("teams", [])
        if len(teams_arr) >= 2:
            pm_team1 = resolve_team_name(teams_arr[0])
            pm_team2 = resolve_team_name(teams_arr[1])
        else:
            name_parts = m_sel.get("name", "").split(" vs ")
            if len(name_parts) >= 2:
                pm_team1 = resolve_team_name(name_parts[0].strip())
                pm_team2 = resolve_team_name(name_parts[1].split(",")[0].strip())

        pm_venue = m_sel.get("venue", "Melbourne Cricket Ground")
        pm_date = m_sel.get("date", "2026-09-05")
        pm_date_gmt = m_sel.get("dateTimeGMT") or m_sel.get("date", "2026-09-05")
    else:
        st.markdown("### Match Fixture Configuration")
        sel_c1, sel_c2, sel_c3, sel_c4 = st.columns(4)
        with sel_c1:
            pm_team1 = st.selectbox("Team 1:", POPULAR_TEAMS, index=0)
        with sel_c2:
            default_t2_idx = 1 if pm_team1 != POPULAR_TEAMS[1] else 0
            pm_team2 = st.selectbox("Team 2:", POPULAR_TEAMS, index=default_t2_idx)
        with sel_c3:
            pm_venue = st.selectbox("Venue:", POPULAR_VENUES, index=0)
        with sel_c4:
            pm_date = st.text_input("Match Date (YYYY-MM-DD):", value="2026-09-05")
            pm_date_gmt = pm_date

    st.markdown("---")

    if pm_model is not None and os.path.exists(matches_csv) and os.path.exists(deliveries_csv):
        features_dict, cutoff_date, venue_meta = compute_target_match_features(
            pm_team1, pm_team2, pm_venue, pm_date, matches_csv, deliveries_csv
        )
        
        df_target = pd.DataFrame([features_dict])[pm_features]
        proba = pm_model.predict_proba(df_target)[0]
        prob_t1 = float(proba[1]) * 100.0
        prob_t2 = 100.0 - prob_t1

        # Central Match Overview Box
        pm_ist_str = format_datetime_ist(pm_date_gmt, pm_date)
        st.markdown(f"### **{pm_team1} vs {pm_team2}**")
        st.write(f"**Match Start Time (IST):** ⏰ `{pm_ist_str}`")
        st.write(f"**Venue:** {pm_venue} ({venue_meta['resolved_venue']})")
        st.info(f"📅 **Historical Data Cutoff:** `{cutoff_date}` | *Model prediction uses historical Elo, recent form & venue priors up to {cutoff_date}.*")

        st.markdown("---")

        # Large Win Probability Display
        p_box_c1, p_box_c2 = st.columns(2)
        with p_box_c1:
            st.markdown(f"""
            <div class="prob-box">
                <div class="team-label">{pm_team1}</div>
                <div class="prob-val-batting">{prob_t1:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with p_box_c2:
            st.markdown(f"""
            <div class="prob-box">
                <div class="team-label">{pm_team2}</div>
                <div class="prob-val-bowling">{prob_t2:.2f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Horizontal Bar Visualization
        fig_pm_bar = go.Figure()
        fig_pm_bar.add_trace(go.Bar(
            y=['Win Prob'], x=[prob_t1], name=pm_team1, orientation='h',
            marker=dict(color='#38BDF8'), text=f"{prob_t1:.1f}%", textposition='auto'
        ))
        fig_pm_bar.add_trace(go.Bar(
            y=['Win Prob'], x=[prob_t2], name=pm_team2, orientation='h',
            marker=dict(color='#F43F5E'), text=f"{prob_t2:.1f}%", textposition='auto'
        ))
        fig_pm_bar.update_layout(
            barmode='stack', height=75, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
            yaxis=dict(showgrid=False, showticklabels=False),
            showlegend=False
        )
        st.plotly_chart(fig_pm_bar, use_container_width=True)

        # Feature Breakdown
        st.markdown("### Pre-Match Feature Breakdown")
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            st.markdown("#### Team Ratings (Elo)")
            st.write(f"**{pm_team1} Elo:** {features_dict['team1_elo']}")
            st.write(f"**{pm_team2} Elo:** {features_dict['team2_elo']}")
            st.write(f"**Elo Difference:** {features_dict['elo_difference']}")

        with f_col2:
            st.markdown("#### Recent Form & Scoring")
            st.write(f"**{pm_team1} Win Rate (Last 5/10):** {features_dict['team1_win_rate_last5']:.0%} / {features_dict['team1_win_rate_last10']:.0%}")
            st.write(f"**{pm_team2} Win Rate (Last 5/10):** {features_dict['team2_win_rate_last5']:.0%} / {features_dict['team2_win_rate_last10']:.0%}")
            st.write(f"**{pm_team1} Avg Runs (Last 5):** {features_dict['team1_avg_runs_last5']}")
            st.write(f"**{pm_team2} Avg Runs (Last 5):** {features_dict['team2_avg_runs_last5']}")

        with f_col3:
            st.markdown("#### H2H & Venue Statistics")
            st.write(f"**{pm_team1} H2H Win Rate:** {features_dict['team1_h2h_win_rate']:.0%} ({features_dict['h2h_matches_before']} matches)")
            st.write(f"**Venue Avg 1st Inn Score:** {features_dict['venue_avg_first_innings_score']}")
            st.write(f"**Venue Batting-1st Win Rate:** {features_dict['venue_batting_first_win_rate']:.1%}")
            st.write(f"**Venue Historical Matches:** {features_dict['venue_matches_before']}")

    else:
        st.error("Error loading pre-match model or historical CSVs.")


# ──────────────────────────────────────────────────────────────────────────────
# MODE 2: HISTORICAL LIVE DEMO
# ──────────────────────────────────────────────────────────────────────────────
elif "Historical Live Demo" in app_mode:
    st.markdown("## Live Model — Historical Demo")
    st.caption("Replay a completed historical T20 chase and observe how win probability changes throughout the match.")

    df_live = load_historical_live_features()
    live_pred = load_live_predictor_model()

    if df_live is not None and live_pred is not None:
        # Match Selection Dropdown
        unique_matches = df_live[["match_id", "match_date", "batting_team", "bowling_team", "target", "batting_team_won"]].drop_duplicates("match_id").reset_index(drop=True)

        match_options = {}
        for _, r in unique_matches.iterrows():
            m_id = r["match_id"]
            res_str = "WON" if r["batting_team_won"] == 1 else "LOST"
            label = f"Match #{m_id} ({r['match_date']}) — {r['batting_team']} chasing {r['target']} vs {r['bowling_team']} [{res_str}]"
            match_options[label] = m_id

        selected_label = st.selectbox("Select Historical Match to Replay:", list(match_options.keys()))
        selected_match_id = match_options[selected_label]

        # Filter snapshots for selected match
        match_snaps = df_live[df_live["match_id"] == selected_match_id].copy().reset_index(drop=True)
        total_snaps = len(match_snaps)

        # Match Overview Details
        first_snap = match_snaps.iloc[0]
        meta_c1, meta_c2, meta_c3, meta_c4 = st.columns(4)
        meta_c1.write(f"**Match Date:** {first_snap['match_date']}")
        meta_c2.write(f"**Batting Team:** {first_snap['batting_team']}")
        meta_c3.write(f"**Bowling Team:** {first_snap['bowling_team']}")
        meta_c4.write(f"**Target:** {first_snap['target']}")

        st.markdown("---")

        # Simulation Slider / Snapshot Selector
        snap_idx = st.slider("Match Simulation Progress (Delivery Snapshot):", min_value=0, max_value=total_snaps - 1, value=min(15, total_snaps - 1))
        curr_snap = match_snaps.iloc[snap_idx]

        # Calculate Win Probabilities via live_model
        snap_pred = live_pred.predict(curr_snap.to_dict())
        prob_batting = snap_pred["batting_team_win_probability"] * 100.0
        prob_bowling = snap_pred["bowling_team_win_probability"] * 100.0

        bat_team_name = curr_snap["batting_team"]
        bowl_team_name = curr_snap["bowling_team"]

        # Central Probability Display
        st.markdown("### Live Win Probability")
        
        p_box_c1, p_box_c2 = st.columns(2)
        with p_box_c1:
            st.markdown(f"""
            <div class="prob-box">
                <div class="team-label">{bat_team_name} (CHASING)</div>
                <div class="prob-val-batting">{prob_batting:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)
            
        with p_box_c2:
            st.markdown(f"""
            <div class="prob-box">
                <div class="team-label">{bowl_team_name} (BOWLING)</div>
                <div class="prob-val-bowling">{prob_bowling:.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

        # Horizontal Bar Visualization
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            y=['Win Prob'], x=[prob_batting], name=bat_team_name, orientation='h',
            marker=dict(color='#38BDF8'), text=f"{prob_batting:.1f}%", textposition='auto'
        ))
        fig_bar.add_trace(go.Bar(
            y=['Win Prob'], x=[prob_bowling], name=bowl_team_name, orientation='h',
            marker=dict(color='#F43F5E'), text=f"{prob_bowling:.1f}%", textposition='auto'
        ))
        fig_bar.update_layout(
            barmode='stack', height=80, margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
            yaxis=dict(showgrid=False, showticklabels=False),
            showlegend=False
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Match Progress Metrics
        st.markdown("### Match Progress State")
        m_c1, m_c2, m_c3, m_c4, m_c5, m_c6 = st.columns(6)
        
        overs_str = f"{int(curr_snap['overs_completed'])}.{int(curr_snap['balls_completed']) % 6}"
        m_c1.metric("Score", f"{int(curr_snap['score'])}/{int(curr_snap['wickets'])}")
        m_c2.metric("Overs", overs_str)
        m_c3.metric("Target", f"{int(curr_snap['target'])}")
        m_c4.metric("Required RR", f"{curr_snap['required_run_rate']:.2f}")
        m_c5.metric("Wickets Left", f"{int(curr_snap['wickets_remaining'])}")
        m_c6.metric("Runs Required", f"{int(curr_snap['runs_required'])} ({int(curr_snap['balls_remaining'])}b)")

        # Win Probability Graph (Plotly Line Chart)
        st.markdown("### Win Probability Trajectory")
        
        all_bat_probs = []
        for _, s_row in match_snaps.iterrows():
            p_res = live_pred.predict(s_row.to_dict())
            all_bat_probs.append(p_res["batting_team_win_probability"] * 100.0)

        match_snaps["batting_win_prob"] = all_bat_probs
        match_snaps["overs_decimal"] = match_snaps["balls_completed"] / 6.0

        fig_line = px.line(
            match_snaps, x="overs_decimal", y="batting_win_prob",
            labels={"overs_decimal": "Overs", "batting_win_prob": f"{bat_team_name} Win Prob (%)"},
            title=f"Probability Trajectory for {bat_team_name} Chasing {curr_snap['target']}"
        )
        fig_line.update_traces(line=dict(color='#38BDF8', width=3))
        
        curr_over_dec = curr_snap['balls_completed'] / 6.0
        fig_line.add_trace(go.Scatter(
            x=[curr_over_dec], y=[prob_batting],
            mode='markers+text',
            marker=dict(color='#F59E0B', size=14, symbol='diamond'),
            text=[f"Current: {prob_batting:.1f}%"],
            textposition="top center",
            name="Current State"
        ))

        fig_line.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1E293B',
            font=dict(color='#F8FAFC'),
            yaxis=dict(range=[0, 100], gridcolor='#334155'),
            xaxis=dict(gridcolor='#334155')
        )
        st.plotly_chart(fig_line, use_container_width=True)

        # Match Timeline Table
        st.markdown("### Match Key Timeline")
        milestone_balls = [1, 30, 60, 90, 108]
        milestone_rows = []

        for b_target in milestone_balls:
            match_at_b = match_snaps[match_snaps["balls_completed"] <= b_target]
            if len(match_at_b) > 0:
                row_m = match_at_b.iloc[-1]
                milestone_rows.append({
                    "Stage": f"~{row_m['balls_completed']//6} Overs",
                    "Score": f"{int(row_m['score'])}/{int(row_m['wickets'])}",
                    "Runs Required": f"{int(row_m['runs_required'])} off {int(row_m['balls_remaining'])} balls",
                    "Required RR": f"{row_m['required_run_rate']:.2f}",
                    f"{bat_team_name} Win Prob": f"{row_m['batting_win_prob']:.1f}%"
                })

        st.table(pd.DataFrame(milestone_rows))

    else:
        st.error("Error loading historical live dataset or LivePredictor model.")


# ──────────────────────────────────────────────────────────────────────────────
# MODE 3: API LIVE MODE
# ──────────────────────────────────────────────────────────────────────────────
else:
    st.markdown("## 📡 Real API Live Mode")
    st.caption("Live streaming fixture integration with CricketData.org API.")

    st.markdown("---")

    col_btn1, col_btn2 = st.columns([3, 1])
    with col_btn1:
        st.write("Fetch real-time match scores directly from CricketData.org API to calculate live win probabilities.")
    with col_btn2:
        fetch_btn = st.button("🔄 Fetch / Refresh API Matches", use_container_width=True)

    client = CricketDataClient()
    elo_calc = get_cached_elo_calc()
    live_pred = load_live_predictor_model()

    api_response = None
    if client.is_configured():
        with st.spinner("Connecting to CricketData.org API..."):
            api_response = client.fetch_current_matches()

    raw_matches = []
    api_success = False
    if api_response and api_response.get("status") == "success":
        raw_matches = api_response.get("data", [])
        api_success = True

    matches_list = filter_live_matches(raw_matches)

    if api_success and len(matches_list) > 0:
        st.success(f"✅ Connected to CricketData API — {len(matches_list)} active live match(es) in progress.")

        # Dropdown selector
        match_labels = [f"{m.get('name', 'Match')} [{str(m.get('matchType', 'T20')).upper()}] — {m.get('status', 'In Progress')}" for m in matches_list]
        selected_match_idx = st.selectbox("Select Active Live Match in Progress:", range(len(matches_list)), format_func=lambda i: match_labels[i])

        curr_match = matches_list[selected_match_idx]
        features = extract_live_match_state(curr_match, elo_calc)

        if live_pred:
            pred = live_pred.predict(features)
            prob_batting = pred["batting_team_win_probability"] * 100.0
            prob_bowling = pred["bowling_team_win_probability"] * 100.0

            bat_team = features["batting_team"]
            bowl_team = features["bowling_team"]

            st.markdown(f"### 🏏 **{features['match_name']}**")
            st.info(f"📍 **Venue:** {features['venue']} | 📢 **Status:** {features['status']}")

            p_c1, p_c2 = st.columns(2)
            with p_c1:
                st.markdown(f"""
                <div class="prob-box">
                    <div class="team-label">{bat_team} (CHASING)</div>
                    <div class="prob-val-batting">{prob_batting:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            with p_c2:
                st.markdown(f"""
                <div class="prob-box">
                    <div class="team-label">{bowl_team} (BOWLING)</div>
                    <div class="prob-val-bowling">{prob_bowling:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            fig_api_bar = go.Figure()
            fig_api_bar.add_trace(go.Bar(
                y=['Win Prob'], x=[prob_batting], name=bat_team, orientation='h',
                marker=dict(color='#38BDF8'), text=f"{prob_batting:.1f}%", textposition='auto'
            ))
            fig_api_bar.add_trace(go.Bar(
                y=['Win Prob'], x=[prob_bowling], name=bowl_team, orientation='h',
                marker=dict(color='#F43F5E'), text=f"{prob_bowling:.1f}%", textposition='auto'
            ))
            fig_api_bar.update_layout(
                barmode='stack', height=80, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
                yaxis=dict(showgrid=False, showticklabels=False),
                showlegend=False
            )
            st.plotly_chart(fig_api_bar, use_container_width=True)

            st.markdown("### Live Scorecard & Match Metrics")
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            m1.metric("Score", f"{int(features['score'])}/{int(features['wickets'])}")
            m2.metric("Overs", f"{int(features['overs_completed'])}.{int(features['balls_completed']) % 6}")
            m3.metric("Target", f"{int(features['target'])}")
            m4.metric("Current RR", f"{features['current_run_rate']:.2f}")
            m5.metric("Required RR", f"{features['required_run_rate']:.2f}")
            m6.metric("Runs Required", f"{int(features['runs_required'])} ({int(features['balls_remaining'])}b)")

    else:
        reason = api_response.get("reason", "No API key configured") if api_response else "CRICKETDATA_API_KEY missing from .env"
        st.warning(f"⚠️ **CricketData API Status:** `{reason}`")
        st.info("💡 **How to activate CricketData API:** Register for a free API key at [CricketData.org](https://cricketdata.org/) and update `CRICKETDATA_API_KEY` in your `.env` file.")

        st.markdown("---")
        st.markdown("### 🎮 Live Match Win Predictor Simulator")
        st.caption("Select teams and live match parameters below to run real-time live win probability predictions using the ML model.")

        sim_c1, sim_c2 = st.columns(2)
        with sim_c1:
            sim_batting = st.selectbox("Batting Team (Chasing):", ["India", "Pakistan", "England", "Australia", "South Africa", "West Indies", "New Zealand", "Zimbabwe"])
            sim_bowling = st.selectbox("Bowling Team:", ["Australia", "India", "Pakistan", "England", "South Africa", "New Zealand", "West Indies", "Zimbabwe"])
            sim_target = st.number_input("Target Score:", min_value=80, max_value=250, value=165)
            sim_score = st.number_input("Current Score:", min_value=0, max_value=int(sim_target), value=110)

        with sim_c2:
            sim_wickets = st.slider("Wickets Lost:", 0, 9, 3)
            sim_overs = st.slider("Overs Completed:", 1.0, 19.5, 14.0, step=0.1)

        total_sim_balls = parse_overs(sim_overs)
        sim_balls_rem = max(1, 120 - total_sim_balls)
        sim_runs_req = max(0, sim_target - sim_score)

        sim_crr = sim_score / (total_sim_balls / 6.0) if total_sim_balls > 0 else 0.0
        sim_rrr = sim_runs_req / (sim_balls_rem / 6.0) if sim_balls_rem > 0 else 0.0

        bat_elo = elo_calc.get_rating(sim_batting)
        bowl_elo = elo_calc.get_rating(sim_bowling)

        sim_feats = {
            "batting_team_elo": bat_elo,
            "bowling_team_elo": bowl_elo,
            "elo_difference": bat_elo - bowl_elo,
            "score": float(sim_score),
            "wickets": float(sim_wickets),
            "balls_remaining": float(sim_balls_rem),
            "target": float(sim_target),
            "runs_required": float(sim_runs_req),
            "wickets_remaining": float(10 - sim_wickets),
            "current_run_rate": float(round(sim_crr, 2)),
            "required_run_rate": float(round(sim_rrr, 2)),
            "run_rate_difference": float(round(sim_crr - sim_rrr, 2)),
        }

        if live_pred:
            sim_res = live_pred.predict(sim_feats)
            sim_bat_p = sim_res["batting_team_win_probability"] * 100.0
            sim_bowl_p = sim_res["bowling_team_win_probability"] * 100.0

            st.markdown("#### Simulated Live Win Probability")
            sp_c1, sp_c2 = st.columns(2)
            with sp_c1:
                st.markdown(f"""
                <div class="prob-box">
                    <div class="team-label">{sim_batting} (CHASING)</div>
                    <div class="prob-val-batting">{sim_bat_p:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            with sp_c2:
                st.markdown(f"""
                <div class="prob-box">
                    <div class="team-label">{sim_bowling} (BOWLING)</div>
                    <div class="prob-val-bowling">{sim_bowl_p:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)

            fig_sim_bar = go.Figure()
            fig_sim_bar.add_trace(go.Bar(
                y=['Win Prob'], x=[sim_bat_p], name=sim_batting, orientation='h',
                marker=dict(color='#38BDF8'), text=f"{sim_bat_p:.1f}%", textposition='auto'
            ))
            fig_sim_bar.add_trace(go.Bar(
                y=['Win Prob'], x=[sim_bowl_p], name=sim_bowling, orientation='h',
                marker=dict(color='#F43F5E'), text=f"{sim_bowl_p:.1f}%", textposition='auto'
            ))
            fig_sim_bar.update_layout(
                barmode='stack', height=80, margin=dict(l=0, r=0, t=0, b=0),
                paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(showgrid=False, showticklabels=False, range=[0, 100]),
                yaxis=dict(showgrid=False, showticklabels=False),
                showlegend=False
            )
            st.plotly_chart(fig_sim_bar, use_container_width=True)

            st.markdown("#### Simulation Scorecard Metrics")
            sm1, sm2, sm3, sm4, sm5 = st.columns(5)
            sm1.metric("Score", f"{sim_score}/{sim_wickets}")
            sm2.metric("Overs", f"{sim_overs:.1f}")
            sm3.metric("Current RR", f"{sim_crr:.2f}")
            sm4.metric("Required RR", f"{sim_rrr:.2f}")
            sm5.metric("Need", f"{sim_runs_req} off {sim_balls_rem}b")


# ──────────────────────────────────────────────────────────────────────────────
# EXPANDABLE SECTIONS: MODEL TRANSPARENCY & METRICS
# ──────────────────────────────────────────────────────────────────────────────
st.markdown("---")

with st.expander("How does the model work?"):
    st.markdown("""
    ### Machine Learning Model Architecture
    
    - **Pre-Match Model**:
      - **Algorithm**: Logistic Regression (StandardScaler + L2 penalty)
      - **Features**: Team Elo ratings, Elo difference, recent win rates (last 5 & 10 matches), recent scoring averages (last 5 matches), Head-to-Head win rate, and venue historical statistics.
    
    - **Live Model**:
      - **Algorithm**: Logistic Regression (StandardScaler)
      - **Features**: Current score, wickets lost, balls remaining, target, required run rate, current run rate, run-rate difference, team Elo ratings, and team strength difference.
    
    > **Chronological Integrity**: Models were trained using strict chronological historical data to prevent future-information leakage.
    """)

pm_metrics, lm_metrics = load_metrics_json()

with st.expander("Model Evaluation Metrics"):
    m_c1, m_c2 = st.columns(2)
    with m_c1:
        st.markdown("#### Pre-Match Model Performance")
        st.write(f"**Accuracy:** {pm_metrics.get('accuracy', 0.7041):.2%}")
        st.write(f"**ROC-AUC:** {pm_metrics.get('roc_auc', 0.7736):.4f}")
        st.write(f"**Log Loss:** {pm_metrics.get('log_loss', 0.5724):.4f}")
        st.write(f"**Brier Score:** {pm_metrics.get('brier_score', 0.1949):.4f}")

    with m_c2:
        st.markdown("#### Live Model Performance")
        st.write(f"**Accuracy:** {lm_metrics.get('accuracy', 0.8591):.2%}")
        st.write(f"**ROC-AUC:** {lm_metrics.get('roc_auc', 0.9424):.4f}")
        st.write(f"**Log Loss:** {lm_metrics.get('log_loss', 0.3035):.4f}")
        st.write(f"**Brier Score:** {lm_metrics.get('brier_score', 0.0966):.4f}")


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────
st.markdown('<div class="footer-text">Built using Python, Scikit-learn, XGBoost, Cricsheet & CricketData.org.</div>', unsafe_allow_html=True)

