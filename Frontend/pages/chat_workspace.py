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
    st.markdown("""
    <style>
        .chat-container { display: flex; flex-direction: column; margin-bottom: 20px; }
        .patient-header {
            background-color: #1E1E1E; border: 1px solid #333; padding: 15px 20px;
            border-radius: 12px; margin-bottom: 20px; display: flex;
            justify-content: space-between; align-items: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }
        .patient-info-group { display: flex; gap: 25px; }
        .patient-stat { display: flex; flex-direction: column; border-left: 1px solid #333; padding-left: 15px; }
        .patient-stat:first-child { border-left: none; padding-left: 0; }
        .stat-label { font-size: 0.75rem; color: #888; text-transform: uppercase; margin-bottom: 2px; }
        .stat-value { font-size: 1.1rem; font-weight: 600; color: #fff; }

        .message-bubble { 
            max-width: 75%; padding: 12px 18px; border-radius: 18px; 
            font-family: 'Segoe UI', sans-serif; line-height: 1.6; 
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .user-message-container { display: flex; justify-content: flex-end; width: 100%; margin-bottom: 15px; }
        .user-message { background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%); color: white; border-bottom-right-radius: 4px; }
        .ai-message-container { display: flex; justify-content: flex-start; width: 100%; margin-bottom: 15px; }
        .ai-message { background-color: #2D2D2D; color: #E0E0E0; border: 1px solid #3D3D3D; border-bottom-left-radius: 4px; }

        .heatmap-container { margin-top: 10px; background-color: #1a1a1a; padding: 10px; border-radius: 12px; text-align: center; border: 1px solid #333; }
        .heatmap-img { max-width: 100%; border-radius: 8px; cursor: pointer; transition: transform 0.2s; }
        .heatmap-img:hover { transform: scale(1.01); }
        .status-success { color: #4CAF50; font-weight: bold; }
        .status-error { color: #F44336; font-weight: bold; }
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
            img_html = f'<div class="heatmap-container"><img src="{heatmap_url}" class="heatmap-img"><br><small>ℹ️ Source: Backend</small></div>'
            st.markdown(
                f'<div class="ai-message-container"><div class="message-bubble ai-message">{img_html}</div></div>',
                unsafe_allow_html=True)


def display_chat_history():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    if not st.session_state.chat_history:
        st.session_state.chat_history.append({
            "role": "model",
            "parts": [{"text": "👋 Hello! Describe symptoms or attach a file."}]
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
    """Извлекает текст из ответа AI в разных форматах"""
    if not ai_response:
        return "⚠️ Empty response."

    # ИСПРАВЛЕНО: Проверяем разные форматы ответа
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
            with st.spinner("AI thinking..."):
                resp = send_chat_message(session_id, user_message)
            add_message_to_history("model", extract_ai_response_text(resp))
        else:
            add_message_to_history("model", "⚠️ API Client is not available.")
    except Exception as e:
        add_message_to_history("model", f"⚠️ Error: {e}")


def handle_file_upload(session_id, patient_id, uploaded_file, file_type, user_symptoms):
    backend_type = ANALYSIS_TYPE_MAP.get(file_type)
    add_message_to_history("user", user_symptoms if user_symptoms else f"🔎 File: {uploaded_file.name}")
    try:
        if analyze_file:
            with st.spinner("Analyzing..."):
                result = analyze_file(patient_id, backend_type, uploaded_file.name, uploaded_file.getvalue(),
                                      user_symptoms or "Analyze this.")
            if result and (backend_type == "ocr" or result.get("heatmap_storage_path")):
                add_message_to_history("system_visualization", "Result", is_file_analysis=True, **result)
                add_message_to_history("model", '<span class="status-success">✅ Analysis complete.</span>')
            else:
                add_message_to_history("model", '<span class="status-error">⚠️ Analysis incomplete.</span>')
        else:
            add_message_to_history("model", "⚠️ API Unavailable.")
    except Exception as e:
        add_message_to_history("model", f"⚠️ Error: {e}")


# MAIN RENDER FUNCTION
def render():
    try:
        inject_custom_css()

        # Проверка инициализации
        if 'file_uploader_key' not in st.session_state: st.session_state.file_uploader_key = 0
        if 'session_id' not in st.session_state:
            st.warning("⚠️ No session active. Please go back to Dashboard.")
            if st.button("⬅️ Dashboard"):
                st.session_state.page = 'dashboard'
                st.rerun()
            return

        patient = st.session_state.get('patient_data', {})
        p_name = patient.get('name', 'Unknown')
        p_id = patient.get('id', 'N/A')
        p_w = patient.get('weight', '-')
        p_h = patient.get('height', '-')

        # Header
        st.markdown(f"""
        <div class="patient-header">
            <div>
                <div class="stat-label">Patient</div>
                <div class="stat-value">{p_name}</div>
            </div>
            <div class="patient-info-group">
                <div class="patient-stat"><div class="stat-label">Weight</div><div class="stat-value">{p_w} kg</div></div>
                <div class="patient-stat"><div class="stat-label">Height</div><div class="stat-value">{p_h} cm</div></div>
                <div class="patient-stat"><div class="stat-label">ID</div><div class="stat-value" style="font-family: monospace;">{p_id}</div></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Button Back
        if st.button("⬅️ Back to List", type="secondary"):
            st.session_state.page = 'dashboard'
            st.rerun()

        display_chat_history()
        st.markdown("---")

        # Input Area
        user_text = st.text_area("Input", placeholder="Type message...", label_visibility="collapsed", height=80,
                                 key="chat_in")
        c1, c2, c3 = st.columns([1.5, 3.5, 1.5])

        uploaded_file = None
        ftype = None

        with c1:
            with st.popover("🔎 Attach"):
                ftype = st.selectbox("Type", list(ANALYSIS_TYPE_MAP.keys()), index=1)
                uploaded_file = st.file_uploader("File", type=["png", "jpg", "pdf", "nii", "gz"],
                                                 key=f"upl_{st.session_state.file_uploader_key}")

        with c3:
            if st.button("Send ➡️", type="primary", use_container_width=True):
                if uploaded_file:
                    handle_file_upload(st.session_state.session_id, p_id, uploaded_file, ftype, user_text)
                    st.session_state.file_uploader_key += 1
                elif user_text:
                    handle_text_message(user_text, st.session_state.session_id)
                st.rerun()

    except Exception as e:
        st.error(f"Critical Render Error: {e}")