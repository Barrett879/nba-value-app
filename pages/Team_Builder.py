"""Team Builder — the contract model from the team's side of the table.

Pick a club and see its whole offseason board: who to re-sign, who to pursue,
the contract it would realistically offer, and why. Same engine as a player's
Likely Suitors, run from the front office's chair. The board itself is rendered
by front_office.render_front_office(); this file is just the page chrome.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import warnings
warnings.filterwarnings("ignore")

import streamlit as st

from utils import render_nav, render_page_chrome, render_footer, _bootstrap_warm
from front_office import render_front_office

st.set_page_config(page_title="Team Builder", page_icon="static/favicon.svg", layout="wide")
render_page_chrome()
_bootstrap_warm()
render_nav("Team Builder")

st.title("Team Builder")
render_front_office()
render_footer()
