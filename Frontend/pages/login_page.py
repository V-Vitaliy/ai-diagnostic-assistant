import streamlit as st
import requests
import os

BASE_API_URL = os.environ.get("BACKEND_URL", "http://backend:8000")

def login_user(email, password):
    url = f"{BASE_API_URL}/auth/token"
    data = {"username": email, "password": password}
    try:
        response = requests.post(url, data=data)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception:
        return None

def register_user(email, password, fullname, role, terms):
    url = f"{BASE_API_URL}/auth/register"
    payload = {
        "email": email,
        "password": password,
        "full_name": fullname,
        "role": role,
        "terms_accepted": terms
    }
    try:
        response = requests.post(url, json=payload)
        if response.status_code == 200:
            return response.json() # Должен быть dict!
        else:
            st.error(f"Błąd rejestracji: {response.text}")
            return None
    except Exception as e:
        st.error(f"Błąd połączenia: {e}")
        return None

def render_login():
    # ... (CSS и разметка без изменений) ...
    # ...
    st.markdown("""
    <style>
        .main { background-color: #0E1117; }
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
        .login-card {
            background: linear-gradient(145deg, #1a1d29 0%, #151820 100%);
            border: 1px solid #2d3139;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            max-width: 450px;
            margin: 60px auto;
            text-align: center;
        }
        .login-title { font-size: 2rem; font-weight: 700; color: #ffffff; margin-bottom: 8px; letter-spacing: 0.5px; }
        .login-subtitle { color: #7c8db5; font-size: 0.9rem; margin-bottom: 30px; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }
        .stTextInput input { background-color: #1a1d29 !important; border: 1px solid #2d3139 !important; border-radius: 8px !important; color: #E0E0E0 !important; }
        .stTextInput input:focus { border-color: #4e4376 !important; box-shadow: 0 0 0 1px #4e4376 !important; }
    </style>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns([1, 2, 1])

    with c2:
        st.markdown("""
        <div class="login-card">
            <div style="font-size: 3rem; margin-bottom: 10px;">🩺</div>
            <div class="login-title">MediChat AI</div>
            <div class="login-subtitle">System Wspomagania Diagnozy</div>
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["Logowanie", "Rejestracja"])

        with tab1:
            email = st.text_input("Email", key="l_email")
            password = st.text_input("Hasło", type="password", key="l_pass")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Zaloguj się", type="primary", use_container_width=True):
                with st.spinner("Weryfikacja..."):
                    token_data = login_user(email, password)
                    if token_data:
                        st.session_state["token"] = token_data["access_token"]
                        st.session_state["user_email"] = email
                        st.session_state["user_role"] = token_data.get("role", "doctor")
                        st.rerun()
                    else:
                        st.error("Błędny email lub hasło")

        with tab2:
            new_email = st.text_input("Email", key="r_email")
            new_fullname = st.text_input("Imię i Nazwisko", key="r_name")
            new_pass = st.text_input("Hasło", type="password", key="r_pass")
            role_choice = st.radio(
                "Jestem:",
                ("Lekarzem (Doctor)","Pacjentem (Patient)"),
                horizontal = True
            )
            role_value = "doctor" if "Lekarzem" in role_choice else "patient"

            terms = st.checkbox("Akceptuję regulamin serwisu")

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("Utwórz konto", type="secondary", use_container_width=True):
                if not terms:
                    st.warning("Musisz zaakceptować regulamin.")
                else:
                    with st.spinner("Tworzenie konta..."):
                        token_data = register_user(new_email, new_pass, new_fullname, role_value, terms)

                        # --- DEBUG START ---
                        st.write(f"Debug token_data type: {type(token_data)}")
                        st.write(f"Debug token_data value: {token_data}")
                        # --- DEBUG END ---

                        if token_data and isinstance(token_data, dict):
                            st.success("Konto utworzone!")
                            st.session_state["token"] = token_data["access_token"]
                            st.session_state["user_email"] = new_email
                            st.session_state["user_role"] = token_data.get("role", role_value)
                            st.rerun()
                        elif token_data is True:
                             st.error("Ошибка: сервер вернул True вместо JSON. Проверьте backend.")
