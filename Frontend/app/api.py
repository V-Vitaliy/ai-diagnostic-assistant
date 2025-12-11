import requests
import os


BASE_API_URL = os.environ.get("BACKEND_URL", "http://backend:8000")
BASE_EXTERNAL_URL = os.environ.get("EXTERNAL_BACKEND_URL", "http://localhost:8080")

def get_full_api_url(path: str) -> str:
    base = BASE_API_URL.rstrip('/')
    path = path.lstrip('/')
    return f"{base}/{path}"

def get_heatmap_url(storage_path: str) -> str:
    if not storage_path: return None
    relative_path = storage_path.replace("app/", "")
    return f"{BASE_EXTERNAL_URL}/{relative_path.lstrip('/')}"



def fetch_patients():
    """Fetches all patients list (GET)"""
    url = get_full_api_url("patients/")
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Błąd pobierania pacjentów: {e}")
        return []

def create_patient(patient_data):
    """Creates a new patient (POST)"""
    url = get_full_api_url("patients/")
    try:
        response = requests.post(url, json=patient_data, timeout=5)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        print(f"Błąd tworzenia pacjenta: {e}")
        return False

def init_chat_session(patient_id):
    """Initializes chat session (POST)"""
    url = get_full_api_url("chat/session/init")
    payload = {"patient_id": patient_id}
    try:
        response = requests.post(url, json=payload, timeout=5)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Błąd inicjalizacji sesji: {e}")
        return None



def send_chat_message(session_id: str, message: str):
    url = get_full_api_url("chat/message")
    payload = {"session_id": session_id, "message": message}
    try:
        response = requests.post(url, json=payload, timeout=6000)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error sending message: {e}")
        raise e

def analyze_file(patient_id: str, analysis_type: str, file_name: str, file_bytes: bytes, symptoms: str):
    url = get_full_api_url("analyze/")
    files = {'image_file': (file_name, file_bytes, 'application/octet-stream')}
    data = {'analysis_type': analysis_type, 'patient_id': patient_id, 'symptoms': symptoms}
    try:
        response = requests.post(url, data=data, files=files, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Error analyzing file: {e}")
        raise e
