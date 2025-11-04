import streamlit as st
from PIL import Image
import io
import requests  # Import the requests library for making API calls

# --- Configuration Constants ---
# Assuming the FastAPI backend is running on the specified service address
API_URL = "http://backend:8080/analyze/"
ANALYSIS_TYPES = ["chest_xray", "unsupported_type"]  # Options available based on backend logic


# -------------------------------

# --- Function to send the POST request to the API ---
def send_analysis_request(api_url, data, files):
    """
    Sends a POST request with image file and form data to the analysis API.
    Returns the JSON response from the API or raises an exception on error.
    """
    try:
        # Making the POST request
        response = requests.post(api_url, data=data, files=files)
        response.raise_for_status()  # Raise an HTTPError for bad responses (4xx or 5xx)
        return response.json()
    except requests.exceptions.HTTPError as e:
        # Handle API error response (e.g., 400 Bad Request, 500 Internal Error)
        error_detail = e.response.json().get('detail', f'Nieznany błąd HTTP: {e.response.text}')
        st.error(f"Błąd Analizy (Status: {e.response.status_code}): {error_detail}")
        return None
    except requests.exceptions.ConnectionError:
        # Handle connection issues (e.g., backend not running)
        st.error(f"Błąd Połączenia: Upewnij się, że backend jest uruchomiony pod adresem: {API_URL}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        st.error(f"Wystąpił nieoczekiwany błąd podczas wysyłania: {e}")
        return None


# ----------------------------------------------------


# Page configuration
st.set_page_config(
    page_title="Przesyłanie Obrazów",  # Image Upload
    layout="centered"
)

# Application header
st.title("Narzędzie do Przesyłania i Analizy Obrazów")  # Image Upload and Analysis Tool
st.markdown(
    "Skonfiguruj parametry analizy i prześlij plik graficzny.")  # Configure analysis parameters and upload a graphic file.

# --- Analysis Form ---
with st.form("analysis_form"):
    st.subheader("Parametry Analizy")  # Analysis Parameters

    # 1. Input for Patient ID (required by backend)
    patient_id_str = st.text_input("Identyfikator Pacjenta (wymagane)", value="12345",
                                   help="Wymagana liczba całkowita.")  # Patient ID (required)

    # 2. Input for Analysis Type (required by backend)
    analysis_type = st.selectbox(
        "Wybierz Typ Analizy",  # Select Analysis Type
        options=ANALYSIS_TYPES,
        index=0,
        help="Wybierz 'chest_xray' dla obsługiwanej analizy."  # Select 'chest_xray' for supported analysis.
    )

    # 3. Input for optional symptoms (optional by backend)
    symptoms = st.text_area("Objawy (opcjonalnie)", height=50,
                            placeholder="Np. kaszel, gorączka (przekazywane do API)")  # Symptoms (optional)

    # 4. File uploader component (required by backend)
    uploaded_file = st.file_uploader(
        "Wybierz plik obrazu",  # Select image file
        type=["png", "jpg", "jpeg"],  # Accepted file formats
        accept_multiple_files=False  # Allow only a single file upload
    )

    # Submission button
    submit_button = st.form_submit_button(label="Wyślij do Analizy 🚀")  # Send for Analysis
# ---------------------

# --- Submission Logic ---
if submit_button:
    if uploaded_file is None:
        st.error(
            "Proszę najpierw przesłać plik obrazu (w sekcji 'Wybierz plik obrazu').")  # Please upload an image file first.
    elif not patient_id_str.isdigit():
        st.error("Identyfikator Pacjenta musi być liczbą całkowitą.")  # Patient ID must be an integer.
    else:
        # Data validation successful, proceed to API call

        # Read the file content as bytes for the requests library
        file_bytes = uploaded_file.getvalue()

        # Prepare the multipart data payload
        files = {
            # The key 'image_file' must match the parameter name in the FastAPI endpoint
            'image_file': (uploaded_file.name, file_bytes, uploaded_file.type)
        }
        data = {
            # The keys must match the parameter names in the FastAPI endpoint
            'analysis_type': analysis_type,
            'patient_id': patient_id_str,  # Send as string
            'symptoms': symptoms
        }

        # Display loading spinner while waiting for response
        with st.spinner(
                'Wysyłanie pliku do backendu i oczekiwanie na analizę...'):  # Sending file to backend and waiting for analysis...

            # Call the new dedicated function
            results = send_analysis_request(API_URL, data, files)

            if results:
                st.success(
                    "Analiza zakończona sukcesem! Otrzymano odpowiedź z API.")  # Analysis completed successfully! Response received from API.

                # Display results in an expander
                # Zmodyfikowany fragment zgodnie z poleceniem: użycie st.json(results)
                with st.expander("Zobacz Szczegółowe Wyniki Analizy (JSON)"):  # See Detailed Analysis Results (JSON)
                    st.json(results) # <--- TUTAJ ZASTOSOWANO st.json()

                # Display the image again for user context
                st.subheader("Przesłany Obraz:")  # Uploaded Image:
                image = Image.open(io.BytesIO(file_bytes))
                st.image(image, use_column_width=True)

# Footer
st.markdown("---")
st.caption("Aplikacja stworzona przy użyciu Streamlit")  # Application created using Streamlit