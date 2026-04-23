import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import streamlit as st
from utils import _bootstrap_warm

# Start warming all season caches the moment the server boots —
# before any user arrives, so the first visitor doesn't pay the cost.
_bootstrap_warm()

st.set_page_config(page_title="Barrett Score", layout="wide", page_icon="🏀")

# ── Hide Streamlit chrome & sidebar nav ────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-left: 0.5rem; padding-right: 0.5rem; max-width: 100%; }
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }
    [data-testid="stToolbar"]        { display: none !important; }
    [data-testid="stDecoration"]     { display: none !important; }
    [data-testid="stStatusWidget"]   { display: none !important; }
    [data-testid="stAppViewerBadge"] { display: none !important; }
    [data-testid="stBottom"]         { display: none !important; }
    [data-testid="stSidebarNav"]     { display: none !important; }
    .viewerBadge_container__r5tak    { display: none !important; }
    .styles_viewerBadge__CvC9N       { display: none !important; }

    /* Clickable nav card */
    a.nav-card {
        display: block;
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 2rem 1.5rem;
        text-align: center;
        text-decoration: none;
        transition: border-color 0.2s, transform 0.15s;
        cursor: pointer;
    }
    a.nav-card:hover {
        border-color: #e63946;
        transform: translateY(-3px);
        text-decoration: none;
    }
    .nav-icon  { font-size: 2.5rem; margin-bottom: 0.75rem; }
    .nav-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.5rem; color: #fff; }
    .nav-desc  { font-size: 0.85rem; color: #aaa; line-height: 1.5; }
    .nav-cta   {
        display: inline-block;
        margin-top: 1.25rem;
        background: #e63946;
        color: #fff !important;
        border-radius: 8px;
        padding: 0.45rem 1.4rem;
        font-weight: 600;
        font-size: 0.85rem;
    }
    a.nav-card:hover .nav-cta { background: #c1121f; }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 0 2.5rem 0;">
    <div style="font-size:3rem; font-weight:800; letter-spacing:-1px; color:#fff;">
        Barrett Score
    </div>
    <div style="font-size:1.1rem; color:#aaa; margin-top:0.5rem;">
        A stat-driven NBA player valuation tool — who's underpaid, overpaid, and available.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Nav cards (entire card is the link) ───────────────────────────────────────
col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.markdown("""
    <a class="nav-card" href="/Rankings" target="_top">
        <div class="nav-icon">🏆</div>
        <div class="nav-title">Rankings</div>
        <div class="nav-desc">Every NBA player ranked by Barrett Score. Filter by team, position, and season going back to 2006.</div>
        <span class="nav-cta">Open Rankings →</span>
    </a>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <a class="nav-card" href="/Salary_Projector" target="_top">
        <div class="nav-icon">💰</div>
        <div class="nav-title">Salary Projector</div>
        <div class="nav-desc">See what every player should earn based on their Barrett Score rank versus their actual contract.</div>
        <span class="nav-cta">Open Projector →</span>
    </a>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <a class="nav-card" href="/Team_Analysis" target="_top">
        <div class="nav-icon">📊</div>
        <div class="nav-title">Team Analysis</div>
        <div class="nav-desc">Aggregate Barrett Scores by team to find the best and worst roster construction in the league.</div>
        <span class="nav-cta">Open Teams →</span>
    </a>
    """, unsafe_allow_html=True)

with col4:
    st.markdown("""
    <a class="nav-card" href="/Free_Agent_Class" target="_top">
        <div class="nav-icon">🆓</div>
        <div class="nav-title">Free Agent Class</div>
        <div class="nav-desc">UFAs, player options, and team options ranked by Barrett Score — a GM's offseason draft board.</div>
        <span class="nav-cta">Open Free Agents →</span>
    </a>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; color:#555; font-size:0.8rem;">
    barrettscore.com &nbsp;·&nbsp; Data from NBA Stats API &nbsp;·&nbsp; Updated daily
</div>
""", unsafe_allow_html=True)
