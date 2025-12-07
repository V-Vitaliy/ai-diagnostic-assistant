import streamlit as st
# Import dashboard module from Pages package (located next to main.py)
from Pages import dashboard

# Page Configuration
st.set_page_config(page_title="Portal Pacjenta", layout="wide")

# State Initialization
if 'page' not in st.session_state:
    st.session_state['page'] = 'dashboard'

# Routing Logic
if st.session_state['page'] == 'dashboard':
    dashboard.render()

elif st.session_state['page'] == 'chat':
    # Placeholder for Chat Interface (will be implemented fully in US-F2)
    st.title("💬 Czat z AI")

    if 'current_patient' in st.session_state:
        patient_name = st.session_state['current_patient'].get('name', 'Nieznany')
        st.success(f"Sesja aktywna dla pacjenta: {patient_name}")
        st.info(f"ID Sesji: {st.session_state.get('session_id')}")

    # Navigation button to go back
    if st.button("Powrót do listy"):
        st.session_state['page'] = 'dashboard'
        st.rerun()