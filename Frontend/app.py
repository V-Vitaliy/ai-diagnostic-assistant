import streamlit as st
from PIL import Image
import io
import requests
import base64  # Import base64 module for encoding images for HTML embedding

# --- Configuration Constants ---
# Assuming the FastAPI backend is running on the specified service address
# CONNECTION NOTE: This address works ONLY within the Docker/Compose network.
# Ensure the 'backend' container is running correctly.
API_URL = "http://backend:8000/analyze/"
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
        # Handle API error response (4xx or 5xx)
        try:
            # Attempt to get JSON detail if available
            error_detail = e.response.json().get('detail', f'Unknown HTTP Error: {e.response.text}')
        except:
            error_detail = f'Unknown HTTP Error: {e.response.text}'

        st.error(f"Analysis Error (Status: {e.response.status_code}): {error_detail}")
        return None
    except requests.exceptions.ConnectionError:
        # Handle connection issues (e.g., backend not running)
        st.error(f"Connection Error: Ensure the backend is running at: {API_URL}")
        return None
    except Exception as e:
        # Catch any other unexpected errors
        st.error(f"An unexpected error occurred during submission: {e}")
        return None


# --- Function to display the image with the heatmap overlay ---
def display_overlay_image(original_image_bytes, heatmap_base64=None):
    """
    Displays the original image and overlays the AI-generated heatmap (if available)
    using custom HTML/CSS for layering.
    """

    # 1. Convert original image bytes to Base64 for data URI
    original_base64 = base64.b64encode(original_image_bytes).decode('utf-8')
    original_src = f"data:image/jpeg;base64,{original_base64}"

    # 2. Set the source for the heatmap
    if heatmap_base64:
        # Assuming the backend returns raw Base64, we add the prefix
        heatmap_src = f"data:image/png;base64,{heatmap_base64}"
    else:
        # Use a transparent image placeholder and show a warning
        heatmap_src = "data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7"
        st.warning("Warning: Heatmap data (`heatmap_base64`) was not found in the backend response.")

    # 3. HTML and CSS for positioning and layering
    # We use position: absolute on the heatmap to place it exactly over the original image.
    # mix-blend-mode: multiply enhances visibility on medical images.
    html_code = f"""
    <div style="position: relative; width: 100%; max-width: 600px; margin: 0 auto;">
        <img src="{original_src}" style="width: 100%; height: auto; display: block; border-radius: 8px;">

        {'<img src="' + heatmap_src + '" style="position: absolute; top: 0; left: 0; width: 100%; height: 100%; opacity: 0.6; mix-blend-mode: multiply; z-index: 10; border-radius: 8px;">' if heatmap_base64 else ''}
    </div>
    """

    st.markdown("### Visualization (Image + Heatmap)")
    st.markdown(html_code, unsafe_allow_html=True)


# ----------------------------------------------------


# Page configuration
st.set_page_config(
    page_title="Image Upload",
    layout="centered"
)

# Application header
st.title("Image Upload and Analysis Tool")
st.markdown("Configure analysis parameters and upload a graphic file.")

# --- Analysis Form ---
with st.form("analysis_form"):
    st.subheader("Analysis Parameters")

    # 1. Input for Patient ID (required by backend)
    patient_id_str = st.text_input("Patient ID (required)", value="12345",
                                   help="Integer required.")

    # 2. Input for Analysis Type (required by backend)
    analysis_type = st.selectbox(
        "Select Analysis Type",
        options=ANALYSIS_TYPES,
        index=0,
        help="Select 'chest_xray' for supported analysis."
    )

    # 3. Input for optional symptoms (optional by backend)
    symptoms = st.text_area("Symptoms (optional)", height=50,
                            placeholder="e.g. cough, fever (passed to API)")

    # 4. File uploader component (required by backend)
    uploaded_file = st.file_uploader(
        "Select image file",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False  # Allow only a single file upload
    )

    # Submission button
    submit_button = st.form_submit_button(label="Send for Analysis 🚀")
# ---------------------

# --- Submission Logic ---
if submit_button:
    if uploaded_file is None:
        st.error("Please upload an image file first.")
    elif not patient_id_str.isdigit():
        st.error("Patient ID must be an integer.")
    else:
        # Data validation successful, proceed to API call
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
        with st.spinner('Sending file to backend and awaiting analysis...'):
            # Call the dedicated request function
            results = send_analysis_request(API_URL, data, files)

            if results:
                st.success("Analysis completed successfully! Response received from API.")

                st.subheader("Analysis Results 📊")

                # --- Structured display of results based on Backend structure ---
                try:
                    patient_id_from_results = results.get('patient_id', 'N/A')
                    analysis_type_from_results = results.get('analysis_type', 'N/A')
                    image_results = results.get('image_analysis_results', {})
                    llm_report = results.get('llm_report', 'No LLM report available.')

                    # Retrieve the Base64 heatmap data from the results
                    heatmap_base64 = image_results.get('heatmap_base64')

                    # 1. Metrics for context
                    col1, col2, col3 = st.columns(3)

                    col1.metric("Patient ID", patient_id_from_results)
                    col2.metric("Analysis Type", analysis_type_from_results.replace('_', ' ').title())

                    # Display confidence metric
                    confidence = image_results.get('confidence')
                    if isinstance(confidence, (int, float)):
                        col3.metric("AI Model Confidence", f"{confidence * 100:.2f}%")
                    else:
                        col3.metric("AI Model Confidence", "N/A")

                    # 2. VISUALIZATION: Original image + Heatmap Overlay
                    display_overlay_image(file_bytes, heatmap_base64)

                    st.markdown("---")  # Separator

                    # 3. AI Model Findings (CheXNet)
                    st.markdown("### AI Model Findings (CheXNet)")
                    finding = image_results.get('finding', 'No primary finding available.')
                    st.info(f"**Primary Finding:** **{finding}**")

                    location = image_results.get('location')
                    if location:
                        st.write(f"**Location:** *{location}*")

                    st.markdown("---")

                    # 4. LLM Report
                    st.markdown("### LLM Generated Report")
                    if llm_report and llm_report != "Report generation is not yet implemented.":
                        st.success(llm_report)
                    else:
                        st.warning(f"LLM Report: *{llm_report}*")
                        st.caption(
                            "Note: The LLM report is currently a placeholder and must be implemented in the backend.")

                    # Optional: Show raw JSON in an expander for debugging
                    with st.expander("View Raw JSON Result (Debugging)"):
                        st.json(results)

                except Exception as e:
                    # Handle errors during result processing
                    st.error(f"Could not process the structure of the backend response. An error occurred: {e}")
                    st.markdown("Here is the raw JSON response for inspection:")
                    st.json(results)
                # --- End of structured display ---

# Footer
st.markdown("---")
st.caption("Application created using Streamlit")