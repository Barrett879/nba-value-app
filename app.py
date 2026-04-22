import streamlit as st

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

    /* Nav card styling */
    .nav-card {
        background: #1a1a2e;
        border: 1px solid #333;
        border-radius: 12px;
        padding: 2rem 1.5rem;
        text-align: center;
        transition: border-color 0.2s;
        height: 100%;
    }
    .nav-card:hover { border-color: #e63946; }
    .nav-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
    .nav-title { font-size: 1.2rem; font-weight: 700; margin-bottom: 0.4rem; }
    .nav-desc  { font-size: 0.85rem; color: #aaa; line-height: 1.4; }

    /* Make Streamlit buttons look like full-width nav triggers */
    div[data-testid="stButton"] > button {
        width: 100%;
        margin-top: 1.2rem;
        background: #e63946;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.55rem 0;
        font-weight: 600;
        font-size: 0.9rem;
        cursor: pointer;
    }
    div[data-testid="stButton"] > button:hover {
        background: #c1121f;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 3rem 0 2rem 0;">
    <div style="font-size:3rem; font-weight:800; letter-spacing:-1px;">
        Barrett Score
    </div>
    <div style="font-size:1.1rem; color:#aaa; margin-top:0.5rem;">
        A stat-driven NBA player valuation tool — who's underpaid, overpaid, and available.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Nav cards ──────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🏆</div>
        <div class="nav-title">Rankings</div>
        <div class="nav-desc">Every NBA player ranked by Barrett Score. Filter by team, position, and season going back to 2006.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Rankings", key="btn_rankings"):
        st.switch_page("pages/Rankings.py")

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">💰</div>
        <div class="nav-title">Salary Projector</div>
        <div class="nav-desc">See what every player should earn based on their Barrett Score rank versus their actual contract.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Projector", key="btn_projector"):
        st.switch_page("pages/Rankings.py")

with col3:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">📊</div>
        <div class="nav-title">Team Analysis</div>
        <div class="nav-desc">Aggregate Barrett Scores by team to find the best and worst roster construction in the league.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Teams", key="btn_teams"):
        st.switch_page("pages/Rankings.py")

with col4:
    st.markdown("""
    <div class="nav-card">
        <div class="nav-icon">🆓</div>
        <div class="nav-title">Free Agent Class</div>
        <div class="nav-desc">UFAs, player options, and team options ranked by Barrett Score — a GM's offseason draft board.</div>
    </div>
    """, unsafe_allow_html=True)
    if st.button("Open Free Agents", key="btn_fa"):
        st.switch_page("pages/Rankings.py")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:3rem; color:#555; font-size:0.8rem;">
    barrettscore.com &nbsp;·&nbsp; Data from NBA Stats API &nbsp;·&nbsp; Updated daily
</div>
""", unsafe_allow_html=True)
