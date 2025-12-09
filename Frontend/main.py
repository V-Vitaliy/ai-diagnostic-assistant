import streamlit as st

# Import dedicated pages
from pages import dashboard
from pages import chat_workspace

# --- Page Configuration ---
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="AI Medical Assistant",
    page_icon="🩺"
)

# --- CSS TO HIDE SIDEBAR ---

st.markdown("""
<style>
    [data-testid="stSidebar"] {display: none;}
    [data-testid="stSidebarNav"] {display: none;}
</style>
""", unsafe_allow_html=True)

# --- State Initialization ---
if 'page' not in st.session_state:
    st.session_state.page = 'dashboard'
if 'patient_data' not in st.session_state:
    st.session_state.patient_data = None
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# --- ROUTING LOGIC ---

if st.session_state['page'] == 'dashboard':
    dashboard.render()

elif st.session_state['page'] == 'chat':
    chat_workspace.render()
