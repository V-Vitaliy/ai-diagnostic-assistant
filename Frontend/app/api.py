import requests
import os

# Get backend URL from Docker environment variables
BASE_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

def fetch_patients():
    """Fetches all patients list (GET)"""
    try:
        response = requests.get(f"{BASE_URL}/patients/")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Błąd pobierania pacjentów: {e}")
        return []

def init_chat_session(patient_id):
    """Initializes chat session (POST)"""
    try:
        payload = {"patient_id": patient_id}
        response = requests.post(f"{BASE_URL}/chat/session/init", json=payload)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Błąd inicjalizacji sesji: {e}")
        return None

def create_patient(patient_data):
    """
    Sends request to create a new patient (POST).
    Expects a dictionary patient_data corresponding to the DB schema.
    """
    try:
        response = requests.post(f"{BASE_URL}/patients/", json=patient_data)
        response.raise_for_status()
        return True
    except requests.RequestException as e:
        # Print server error text for debugging
        error_text = e.response.text if e.response else str(e)
        print(f"Błąd tworzenia pacjenta: {error_text}")
        return False