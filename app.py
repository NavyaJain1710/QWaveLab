"""
QWaveLab – Interactive Quantum Wave Packet Simulation Laboratory
Streamlit entry point:  streamlit run app.py
"""
import streamlit as st

st.set_page_config(page_title="QWaveLab – Quantum Wave Packet Laboratory", page_icon="⚛️", layout="wide",
                   initial_sidebar_state="expanded")

import pages
import ui

ui.inject_css()

ROUTES = {
    pages.PAGE_NAMES[0]: pages.page_home,
    pages.PAGE_NAMES[1]: pages.page_simulator,
    pages.PAGE_NAMES[2]: pages.page_potential_lab,
    pages.PAGE_NAMES[3]: pages.page_tunnelling,
    pages.PAGE_NAMES[4]: pages.page_resonant,
    pages.PAGE_NAMES[5]: pages.page_momentum,
    pages.PAGE_NAMES[6]: pages.page_classical,
    pages.PAGE_NAMES[7]: pages.page_analysis,
    pages.PAGE_NAMES[8]: pages.page_validation,
}

with st.sidebar:
    st.markdown('<div class="qw-brand">⚛ QWaveLab</div><div class="qw-brand-sub">Quantum wave packet laboratory</div>',
                unsafe_allow_html=True)
    page = st.radio("Navigate", pages.PAGE_NAMES, key="nav", label_visibility="collapsed")
    st.divider()
    st.caption("Units: ħ = 1, m = 1 (dimensionless) unless a page lets you change m.  \n"
               "Solver: Crank–Nicolson, sparse LU, Dirichlet walls.")

# leaving a page stops any running animation
if st.session_state.get("_last_page") != page:
    for k in list(st.session_state.keys()):
        if k.endswith("_play"):
            st.session_state[k] = False
    st.session_state["_last_page"] = page

ROUTES[page]()
