import streamlit as st
from PIL import Image
import io

# Page configuration
st.set_page_config(
    page_title="Przesyłanie Obrazów", # Image Upload
    layout="centered"
)

# Application header
st.title("Narzędzie do Przesyłania Obrazów") # Image Upload Tool
st.markdown("Proszę, prześlij plik graficzny w formacie PNG, JPG lub JPEG.") # Please upload a graphic file in PNG, JPG or JPEG format.

# --- st.file_uploader COMPONENT ---
uploaded_file = st.file_uploader(
    "Wybierz plik obrazu", # Select image file
    type=["png", "jpg", "jpeg"],  # Accepted file formats
    accept_multiple_files=False    # Allow only a single file upload
)
# -----------------------------------

if uploaded_file is not None:
    try:
        # Successful file upload
        st.success(f"Plik '{uploaded_file.name}' został pomyślnie przesłany!") # File successfully uploaded!

        # Load the image into a PIL (Python Imaging Library) object
        # For subsequent processing or display
        image = Image.open(io.BytesIO(uploaded_file.read()))

        # Display image information
        st.subheader("Podgląd i Właściwości Obrazu:") # Image Preview and Properties:
        st.image(
            image,
            caption=f"Rozmiar: {image.width}x{image.height} px | Format: {image.format}", # Size: ... | Format: ...
            use_column_width=True
        )

    except Exception as e:
        # Error during file reading
        st.error(f"Wystąpił błąd podczas wczytywania pliku: {e}") # An error occurred while loading the file

else:
    # Message when the file is not yet uploaded
    st.info("Oczekuję na przesłanie pliku. Użyj przycisku powyżej.") # Waiting for file upload. Use the button above.

# Footer
st.markdown("---")
st.caption("Aplikacja stworzona przy użyciu Streamlit") # Application created using Streamlit
