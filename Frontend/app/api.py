import requests
import os
import streamlit as st
import time

BASE_API_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
BASE_EXTERNAL_URL = os.environ.get("EXTERNAL_BACKEND_URL", "http://localhost:8000")

def get_full_api_url(path: str) -> str:
    base = BASE_API_URL.rstrip('/')
    path = path.lstrip('/')
    return f"{base}/{path}"

def get_heatmap_url(storage_path: str) -> str:
    """
    Converts a backend storage path to a browser-accessible URL.
    """
    if not storage_path:
        return None

    # Normalize path separators
    storage_path = storage_path.replace("\\", "/")

    # Extract filename from path
    if "heatmaps/" in storage_path:
        filename = storage_path.split("heatmaps/")[-1]
        base = BASE_EXTERNAL_URL.rstrip('/')
        final_url = f"{base}/static/heatmaps/{filename}"
        return final_url

    if storage_path.endswith("_heatmap.png") or storage_path.endswith(".png"):
        filename = storage_path.split("/")[-1]
        base = BASE_EXTERNAL_URL.rstrip('/')
        final_url = f"{base}/static/heatmaps/{filename}"
        return final_url

    if "/static/" in storage_path:
        _, relative_path = storage_path.split("/static/", 1)
        base = BASE_EXTERNAL_URL.rstrip('/')
        final_url = f"{base}/static/{relative_path}"
        return final_url

    return None

def get_auth_headers():
    token = st.session_state.get("token")
    if token:
        return {"Authorization": f"Bearer {token}"}
    return {}

def fetch_patients():
    url = get_full_api_url("patients/")
    try:
        response = requests.get(url, headers=get_auth_headers(), timeout=5)
        if response.status_code == 401:
            st.error("Sesja wygasła. Zaloguj się ponownie.")
            return []
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error fetching patients: {e}")
        return []

def create_patient(patient_data):
    url = get_full_api_url("patients/")
    try:
        response = requests.post(url, json=patient_data, headers=get_auth_headers(), timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Error creating patient: {e}")
        return False

def init_chat_session(patient_id):
    url = get_full_api_url("chat/session/init")
    payload = {"patient_id": patient_id}
    try:
        response = requests.post(url, json=payload, headers=get_auth_headers(), timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error initializing session: {e}")
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

# --- NEW: Polling Functions ---

def get_analysis_status(analysis_id: str):
    """Checks the status of a specific analysis."""
    url = get_full_api_url(f"analyze/{analysis_id}")
    try:
        headers = get_auth_headers()
        response = requests.get(url, headers=headers, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error checking status: {e}")
        return {"status": "ERROR", "error": str(e)}

def analyze_file(patient_id: str, session_id: str, analysis_type: str, file_name: str, file_bytes: bytes, symptoms: str):
    """
    Initiates analysis (returns immediately with ID) and does NOT wait for completion.
    """
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
        # Short timeout for initial handshake (we just get ID back)
        response = requests.post(url, data=data, files=files, headers=headers, timeout=30)
        response.raise_for_status()
        return response.json() # Returns {"id": "...", "status": "PENDING"}
    except requests.RequestException as e:
        print(f"Error initiating analysis: {e}")
        raise e