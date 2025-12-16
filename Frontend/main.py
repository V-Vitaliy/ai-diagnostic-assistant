import streamlit as st

# --- 1. Page Config MUST be first ---
st.set_page_config(
    layout="wide",
    initial_sidebar_state="collapsed",
    page_title="AI Medical Assistant",
    page_icon="🩺"
)

# Import pages
from pages import dashboard
from pages import chat_workspace
from pages import login_page

# --- 2. Auth Logic ---
if "token" not in st.session_state:
    login_page.render_login()
else:
    # --- 3. App Logic (Only if logged in) ---

    # Global CSS for Sidebar and consistent theme
    st.markdown("""
    <style>
        /* Hide default sidebar nav items */
        [data-testid="stSidebarNav"] {display: none;}
        
        /* Sidebar container styling to match dashboard theme */
        [data-testid="stSidebar"] {
            background-color: #0E1117;
            border-right: 1px solid #2d3139;
        }
        
        /* Sidebar User Profile Card - Matches .patient-card style */
        .sidebar-card {
            background: linear-gradient(145deg, #1a1d29 0%, #151820 100%);
            border: 1px solid #2d3139;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            margin-bottom: 24px;
        }
        
        .sidebar-avatar {
            font-size: 3rem;
            margin-bottom: 12px;
            text-shadow: 0 2px 10px rgba(0,0,0,0.5);
        }
        
        .sidebar-user-email {
            color: #ffffff;
            font-weight: 700;
            font-size: 0.95rem;
            margin-bottom: 4px;
            word-wrap: break-word;
            letter-spacing: 0.3px;
        }
        
        .sidebar-label {
            color: #7c8db5;
            font-size: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            font-weight: 600;
        }
        
        /* Custom separator */
        .sidebar-divider {
            height: 1px;
            background-color: #2d3139;
            margin: 20px 0;
        }
    </style>
    """, unsafe_allow_html=True)

    # Init State
    if 'page' not in st.session_state:
        st.session_state.page = 'dashboard'
    if 'patient_data' not in st.session_state:
        st.session_state.patient_data = None
    if 'session_id' not in st.session_state:
        st.session_state.session_id = None
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Routing
    if st.session_state['page'] == 'dashboard':
        dashboard.render()
    elif st.session_state['page'] == 'chat':
        chat_workspace.render()

    # Custom Sidebar with Logout
    with st.sidebar:
        user_email = st.session_state.get('user_email', 'Unknown')
        user_role = st.session_state.get('user_role', 'doctor')

        # Render custom profile card
        role_label = "Lekarz 👨‍⚕️" if user_role == "doctor" else "Pacjent 👤"
        st.markdown(f"""
        <div class="sidebar-card">
            <div class="sidebar-avatar">{"👨‍⚕️" if user_role == "doctor" else "👤"}</div>
            <div class="sidebar-label">{role_label}</div>
            <div class="sidebar-user-email">{user_email}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚪 Wyloguj", type="primary", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()