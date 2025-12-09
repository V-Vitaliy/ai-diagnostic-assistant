import streamlit as st
import os
import re
import sys
from requests.exceptions import HTTPError, ConnectionError

from app.api import send_chat_message, analyze_file, get_heatmap_url, BASE_API_URL

ANALYSIS_TYPE_MAP = {
    "Foto analizy krwi/moczu": "ocr",
    "Rentgen klatki piersiowej": "chest_xray",
    "Tomografia komputerowa (3D)": "whole_body_ct",
    "Rentgen kończyn/kości": "extremity_xray"
}


def inject_custom_css():
    """Unified CSS styling matching dashboard design"""
    st.markdown("""
    <style>
        /* Global styling */
        .main { background-color: #0E1117; }

        /* Patient header - matching dashboard cards */
        .patient-header {
            background: linear-gradient(145deg, #1a1d29 0%, #151820 100%);
            border: 1px solid #2d3139;
            padding: 20px 24px;
            border-radius: 16px;
            margin-bottom: 24px;
            display: flex;
            justify-content: space-between;
            align-items: center;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
        }

        .patient-main-info {
            display: flex;
            flex-direction: column;
        }

        .patient-name-large {
            font-size: 1.5rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.3px;
            margin-bottom: 4px;
        }

        .patient-id-label {
            color: #7c8db5;
            font-size: 0.8rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }

        .patient-info-group { 
            display: flex; 
            gap: 28px; 
        }

        .patient-stat { 
            display: flex; 
            flex-direction: column; 
            border-left: 1px solid #2d3139; 
            padding-left: 18px; 
        }

        .patient-stat:first-child { 
            border-left: none; 
            padding-left: 0; 
        }

        .stat-label { 
            font-size: 0.75rem; 
            color: #7c8db5; 
            text-transform: uppercase; 
            margin-bottom: 4px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }

        .stat-value { 
            font-size: 1.15rem; 
            font-weight: 600; 
            color: #ffffff; 
        }

        /* Chat messages */
        .chat-container { 
            display: flex; 
            flex-direction: column; 
            margin-bottom: 20px; 
        }

        .message-bubble { 
            max-width: 75%; 
            padding: 14px 20px; 
            border-radius: 18px; 
            font-family: 'Segoe UI', sans-serif; 
            line-height: 1.6; 
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
            font-size: 0.95rem;
        }

        .user-message-container { 
            display: flex; 
            justify-content: flex-end; 
            width: 100%; 
            margin-bottom: 16px; 
        }

        .user-message { 
            background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%); 
            color: white; 
            border-bottom-right-radius: 4px;
        }

        .ai-message-container { 
            display: flex; 
            justify-content: flex-start; 
            width: 100%; 
            margin-bottom: 16px; 
        }

        .ai-message { 
            background: linear-gradient(145deg, #1a1d29 0%, #151820 100%);
            color: #E0E0E0; 
            border: 1px solid #2d3139; 
            border-bottom-left-radius: 4px;
        }

        /* Heatmap container */
        .heatmap-container { 
            margin-top: 12px; 
            background: linear-gradient(145deg, #1a1d29 0%, #151820 100%);
            padding: 14px; 
            border-radius: 12px; 
            text-align: center; 
            border: 1px solid #2d3139;
            box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        }

        .heatmap-img { 
            max-width: 100%; 
            border-radius: 8px; 
            cursor: pointer; 
            transition: transform 0.2s; 
        }

        .heatmap-img:hover { 
            transform: scale(1.02); 
        }

        /* Status messages */
        .status-success { 
            color: #4CAF50; 
            font-weight: bold; 
        }

        .status-error { 
            color: #F44336; 
            font-weight: bold; 
        }

        /* Input area styling */
        .stTextArea textarea {
            background-color: #1a1d29 !important;
            border: 1px solid #2d3139 !important;
            border-radius: 12px !important;
            color: #E0E0E0 !important;
        }

        .stTextArea textarea:focus {
            border-color: #4e4376 !important;
            box-shadow: 0 0 0 1px #4e4376 !important;
        }

        /* Custom divider */
        hr {
            border-color: #2d3139;
            margin: 24px 0;
        }
    </style>
    """, unsafe_allow_html=True)


def format_markdown(text):
    if not text: return ""
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'^\s*[\*-]\s+(.+)$', r'• \1', text, flags=re.MULTILINE)
    return text


def display_message_html(role, content, heatmap_url=None):
    formatted_content = format_markdown(content)
    if role == "user":
        st.markdown(
            f'<div class="user-message-container"><div class="message-bubble user-message">{formatted_content}</div></div>',
            unsafe_allow_html=True)
    elif role == "model" or role == "assistant":
        st.markdown(
            f'<div class="ai-message-container"><div class="message-bubble ai-message">{formatted_content}</div></div>',
            unsafe_allow_html=True)
    elif role == "system_visualization":
        if heatmap_url:
            img_html = f'<div class="heatmap-container"><img src="{heatmap_url}" class="heatmap-img"><br><small style="color: #7c8db5;">ℹ️ Wizualizacja z analizy</small></div>'
            st.markdown(
                f'<div class="ai-message-container"><div class="message-bubble ai-message">{img_html}</div></div>',
                unsafe_allow_html=True)


def display_chat_history():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "model",
            "parts": [{"text": "👋 Witaj! Opisz objawy lub dołącz plik do analizy."}]
        })

    for message in st.session_state.chat_history:
        role = message.get("role", "model")
        content = message.get("parts", [{}])[0].get("text", "")
        if role == "user":
            display_message_html("user", content)
        elif role == "system_visualization":
            heatmap_path = message.get('heatmap_storage_path')
            full_url = get_heatmap_url(heatmap_path) if heatmap_path else None
            display_message_html("system_visualization", "", full_url)
        else:
            display_message_html("model", content)


def add_message_to_history(role, text, is_file_analysis=False, **kwargs):
    new_message = {"role": role, "parts": [{"text": text}]}
    if is_file_analysis:
        new_message['role'] = 'system_visualization'
        new_message.update(kwargs)
    st.session_state.chat_history.append(new_message)


def extract_ai_response_text(ai_response):

    if not ai_response:
        return "⚠️ Pusta odpowiedź."

    if "response" in ai_response:
        return ai_response["response"]
    if "parts" in ai_response and ai_response["parts"]:
        return ai_response["parts"][0].get("text", "")
    if "text" in ai_response:
        return ai_response["text"]

    return str(ai_response)


def handle_text_message(user_message, session_id):
    add_message_to_history("user", user_message)
    try:
        if send_chat_message:
            with st.spinner("AI analizuje..."):
                resp = send_chat_message(session_id, user_message)
            add_message_to_history("model", extract_ai_response_text(resp))
        else:
            add_message_to_history("model", "⚠️ Klient API niedostępny.")
    except Exception as e:
        add_message_to_history("model", f"⚠️ Błąd: {e}")


def handle_file_upload(session_id, patient_id, uploaded_file, file_type, user_symptoms):
    backend_type = ANALYSIS_TYPE_MAP.get(file_type)
    add_message_to_history("user", user_symptoms if user_symptoms else f"📎 Plik: {uploaded_file.name}")
    try:
        if analyze_file:
            with st.spinner("Analizuję..."):
                result = analyze_file(patient_id, backend_type, uploaded_file.name, uploaded_file.getvalue(),
                                      user_symptoms or "Przeanalizuj to.")
            if result and (backend_type == "ocr" or result.get("heatmap_storage_path")):
                add_message_to_history("system_visualization", "Wynik analizy", is_file_analysis=True, **result)
                add_message_to_history("model", '<span class="status-success">✅ Analiza zakończona.</span>')
            else:
                add_message_to_history("model", '<span class="status-error">⚠️ Analiza niekompletna.</span>')
        else:
            add_message_to_history("model", "⚠️ API niedostępne.")
    except Exception as e:
        add_message_to_history("model", f"⚠️ Błąd: {e}")


# MAIN RENDER FUNCTION
def render():
    try:
        inject_custom_css()

        if 'file_uploader_key' not in st.session_state:
            st.session_state.file_uploader_key = 0

        if 'session_id' not in st.session_state:
            st.warning("⚠️ Brak aktywnej sesji. Wróć do panelu pacjentów.")
            if st.button("⬅️ Panel pacjentów"):
                st.session_state.page = 'dashboard'
                st.rerun()
            return

        patient = st.session_state.get('patient_data', {})
        p_name = patient.get('name', 'Nieznany')
        p_id = patient.get('id', 'N/A')
        p_w = patient.get('weight_kg', '-')
        p_h = patient.get('height_cm', '-')

        # Header with patient info
        st.markdown(f"""
        <div class="patient-header">
            <div class="patient-main-info">
                <div class="patient-name-large">{p_name}</div>
                <div class="patient-id-label">ID: {p_id}</div>
            </div>
            <div class="patient-info-group">
                <div class="patient-stat">
                    <div class="stat-label">Waga</div>
                    <div class="stat-value">{p_w} kg</div>
                </div>
                <div class="patient-stat">
                    <div class="stat-label">Wzrost</div>
                    <div class="stat-value">{p_h} cm</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Back button
        if st.button("⬅️ Powrót do listy", type="secondary"):
            st.session_state.page = 'dashboard'
            st.rerun()

        # Chat history
        display_chat_history()

        st.markdown("---")

        # Input area
        user_text = st.text_area("", placeholder="Wpisz wiadomość...", label_visibility="collapsed", height=85,
                                 key="chat_in")
        c1, c2, c3 = st.columns([1.5, 3.5, 1.5])

        uploaded_file = None
        ftype = None

        with c1:
            with st.popover("📎 Dołącz plik"):
                ftype = st.selectbox("Typ analizy", list(ANALYSIS_TYPE_MAP.keys()), index=1)
                uploaded_file = st.file_uploader("Wybierz plik", type=["png", "jpg", "pdf", "nii", "gz"],
                                                 key=f"upl_{st.session_state.file_uploader_key}")

        with c3:
            if st.button("Wyślij ➡️", type="primary", use_container_width=True):
                if uploaded_file:
                    handle_file_upload(st.session_state.session_id, p_id, uploaded_file, ftype, user_text)
                    st.session_state.file_uploader_key += 1
                elif user_text:
                    handle_text_message(user_text, st.session_state.session_id)
                st.rerun()

    except Exception as e:
        st.error(f"Błąd krytyczny renderowania: {e}")