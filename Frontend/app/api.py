import requests
import os
import streamlit as st

BASE_API_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
BASE_EXTERNAL_URL = os.environ.get("EXTERNAL_BACKEND_URL", "http://localhost:8000")

def get_full_api_url(path: str) -> str:
    base = BASE_API_URL.rstrip('/')
    path = path.lstrip('/')
    return f"{base}/{path}"

def get_heatmap_url(storage_path: str) -> str:
    """
    Converts a backend storage path to a browser-accessible URL.
    Backend stores: /app/app/static/heatmaps/xyz.png
    Browser needs: http://localhost:8000/static/heatmaps/xyz.png
    """
    if not storage_path:
        return None

    # Normalize path separators
    storage_path = storage_path.replace("\\", "/")

    # We need to find the part of the path starting from 'static/'
    # Because FastAPI usually mounts the static folder to '/static' path.
    if "/static/" in storage_path:
        # Split by the first occurrence of /static/
        # e.g., /app/app/static/heatmaps/img.png -> heatmaps/img.png
        _, relative_path = storage_path.split("/static/", 1)

        # Construct full URL
        # Ensure BASE_EXTERNAL_URL doesn't end with slash
        base = BASE_EXTERNAL_URL.rstrip('/')
        return f"{base}/static/{relative_path}"

    return None

def get_auth_headers():
    """Gets Token from the Streamlit session state and returns Authorization header with Bearer token."""
    token = st.session_state.get("token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def fetch_patients():
    """Fetches all patients list (GET) WITH AUTH"""
    url = get_full_api_url("patients/")
    try:
        response = requests.get(url, headers=get_auth_headers(), timeout=5)

        if response.status_code == 401:
            st.error("Sesja wygasła. Zaloguj się ponownie.")
            return []

        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Błąd pobierania pacjentów: {e}")
        return []

def create_patient(patient_data):
    """Creates a new patient (POST) WITH AUTH"""
    url = get_full_api_url("patients/")
    try:
        response = requests.post(url, json=patient_data, headers=get_auth_headers(), timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Błąd tworzenia pacjenta: {e}")
        return False

def init_chat_session(patient_id):
    """Initializes chat session (POST) WITH AUTH"""
    url = get_full_api_url("chat/session/init")
    payload = {"patient_id": patient_id}
    try:
        response = requests.post(url, json=payload, headers=get_auth_headers(), timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Błąd inicjalizacji sesji: {e}")
        return None

def send_chat_message(session_id: str, message: str):
    url = get_full_api_url("chat/message")
    payload = {"session_id": session_id, "message": message}
    try:
        response = requests.post(url, json=payload, headers=get_auth_headers(), timeout=6000)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error sending message: {e}")
        raise e

def analyze_file(patient_id: str, session_id: str, analysis_type: str, file_name: str, file_bytes: bytes, symptoms: str):
    url = get_full_api_url("analyze/")

    files = {'image_file': (file_name, file_bytes, 'application/octet-stream')}

    data = {
        'analysis_type': analysis_type,
        'patient_id': patient_id,
        'session_id': session_id,
        'symptoms': symptoms
    }

    try:
        headers = get_auth_headers()

        response = requests.post(url, data=data, files=files, headers=headers, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error analyzing file: {e}")
        raise e