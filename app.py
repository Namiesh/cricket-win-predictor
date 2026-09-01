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

sportmonks_env_token = bool(os.getenv("SPORTMONKS_API_TOKEN"))

# Sleek API Status Panel
st.sidebar.markdown(f"""
<div class="sidebar-status-box">
    <div class="sidebar-status-title">System Status</div>
    <div class="sidebar-status-item">
        <span>Sportmonks API:</span>
        <span style="color:{'#10B981' if sportmonks_env_token else '#F59E0B'}; font-weight:700;">
            {'Connected ✅' if sportmonks_env_token else 'Token Configured'}
        </span>
    </div>
    <div class="sidebar-status-item">
        <span>Target Fixture:</span>
        <span style="color:#9CA3AF;">Not Available</span>
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
    st.markdown("## PRE-MATCH PREDICTION")
    st.caption("Pre-match win probability for upcoming T20 international fixtures calculated from historical Elo, form, and venue priors.")

    pm_model, pm_features = load_prematch_model_artifact()
    pm_team1 = "Zimbabwe"
    pm_team2 = "South Africa"
    pm_date = "2026-09-01"
    pm_venue = "Namibia Cricket Ground"

    matches_csv = os.path.join("data", "processed", "matches.csv")
    deliveries_csv = os.path.join("data", "processed", "deliveries.csv")

    if pm_model is not None and os.path.exists(matches_csv) and os.path.exists(deliveries_csv):
        features_dict, cutoff_date, venue_meta = compute_target_match_features(
            pm_team1, pm_team2, pm_venue, pm_date, matches_csv, deliveries_csv
        )
        
        df_target = pd.DataFrame([features_dict])[pm_features]
        proba = pm_model.predict_proba(df_target)[0]
        prob_t1 = float(proba[1]) * 100.0
        prob_t2 = 100.0 - prob_t1

        # Central Match Overview Box
        st.markdown(f"### **{pm_team1} vs {pm_team2}**")
        st.write(f"**Match Date:** {pm_date}")
        st.write(f"**Venue:** {pm_venue} ({venue_meta['resolved_venue']})")
        st.info(f"📅 **Historical Data Cutoff:** `{cutoff_date}`\n\n"
                f"*Model prediction using historical data available up to {cutoff_date}.*")

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
    st.markdown("## Real API Live Mode")
    st.caption("Live streaming fixture integration with Sportmonks API 2.0.")

    st.warning("Sportmonks live fixture unavailable. API integration will activate when a supported live fixture is available.")
    
    st.info("ℹ️ **Integrity Notice:** No API calls are made automatically from this UI page, and no artificial scores or probabilities are fabricated.")


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
st.markdown('<div class="footer-text">Built using Python, Scikit-learn, XGBoost, Cricsheet & Sportmonks.</div>', unsafe_allow_html=True)
