import streamlit as st
from datetime import datetime

from app.api import fetch_patients, init_chat_session, create_patient


def calculate_age(birth_date_str):
    """Calculates age based on date of birth"""
    if not birth_date_str: return "N/A"
    try:
        birth = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.today()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except:
        return "?"


def handle_click(patient):
    """Handler for patient selection: starts session and redirects to chat"""
    with st.spinner(f"Ładowanie karty pacjenta {patient['name']}..."):
        # Инициализируем сессию
        session_data = init_chat_session(patient['id'])

        if session_data and 'session_id' in session_data:
            st.session_state['session_id'] = session_data['session_id']

            # ИСПРАВЛЕНО: читаем из правильного ключа history_json
            history = session_data.get('history_json', [])

            st.session_state['chat_history'] = history

            # Сохраняем данные пациента
            st.session_state['current_patient'] = patient
            st.session_state['patient_data'] = patient

            # Переключаем страницу
            st.session_state['page'] = 'chat'
            st.rerun()
        else:
            st.error("Błąd serwera: nie udało się utworzyć sesji lub pobrać historii.")


@st.dialog("Nowy pacjent")
def open_create_modal():
    st.write("Wypełnij dane, aby utworzyć nową kartę.")

    with st.form("create_patient_form"):
        name = st.text_input("Imię i Nazwisko", placeholder="Jan Kowalski")
        birth_date = st.date_input("Data urodzenia", min_value=datetime(1900, 1, 1), value=datetime(1990, 1, 1))

        # ДОБАВЛЕНО: Поля роста и веса
        col_hw1, col_hw2 = st.columns(2)
        with col_hw1:
            height = st.number_input("Wzrost (cm)", min_value=50, max_value=250, value=170, step=1)
        with col_hw2:
            weight = st.number_input("Waga (kg)", min_value=20.0, max_value=300.0, value=70.0, step=0.5, format="%.1f")

        col1, col2 = st.columns(2)
        with col1:
            chronic = st.text_area("Choroby przewlekłe", placeholder="Cukrzyca, Astma (oddzielone przecinkami)")
        with col2:
            allergies = st.text_area("Alergie", placeholder="Pyłki, Orzechy (oddzielone przecinkami)")

        medications = st.text_input("Przyjmowane leki", placeholder="Aspiryna, Insulina (oddzielone przecinkami)")

        submitted = st.form_submit_button("Zapisz profil", type="primary", use_container_width=True)

        if submitted:
            if not name:
                st.error("Proszę podać imię pacjenta.")
            else:
                def parse_list(text):
                    if not text: return []
                    return [item.strip() for item in text.split(',') if item.strip()]

                new_patient_data = {
                    "name": name,
                    "birth_date": str(birth_date),
                    "chronic_diseases": parse_list(chronic),
                    "allergies": parse_list(allergies),
                    "medications": parse_list(medications),
                    "height_cm": int(height),
                    "weight_kg": float(weight)
                }

                if create_patient(new_patient_data):
                    st.success("Pacjent został pomyślnie utworzony!")
                    st.rerun()
                else:
                    st.error("Nie udało się zapisać pacjenta. Sprawdź logi.")


def render():
    st.title("Wybór Pacjenta")

    patients = fetch_patients()

    cols = st.columns(3)

    # --- HELPER TO DISPLAY LISTS ---
    def display_list_items(label, items):
        if items and isinstance(items, list) and len(items) > 0:
            text = ", ".join(items)
            if len(text) > 40:
                text = text[:37] + "..."
            st.markdown(f"**{label}:** <span style='color: #cccccc; font-size: 0.9em;'>{text}</span>",
                        unsafe_allow_html=True)

    # Loop through existing patients
    for idx, patient in enumerate(patients):
        with cols[idx % 3]:
            with st.container(border=True):
                # Header
                st.subheader(patient.get('name', 'Bez nazwy'))

                # Age
                age = calculate_age(patient.get('birth_date'))
                st.write(f"**Wiek:** {age} lat")

                st.divider()  # Separator line

                # --- MEDICAL INFO SECTIONS ---
                chronic = patient.get('chronic_diseases', [])
                allergies = patient.get('allergies', [])
                medications = patient.get('medications', [])

                has_data = False

                if chronic:
                    display_list_items("Choroby", chronic)
                    has_data = True

                if allergies:
                    display_list_items("Alergie", allergies)
                    has_data = True

                if medications:
                    display_list_items("Leki", medications)
                    has_data = True

                if not has_data:
                    st.caption("Brak danych medycznych")

                st.write("")

                # Кнопка перехода к чату
                if st.button("Otwórz kartę", key=f"btn_{patient['id']}", use_container_width=True):
                    handle_click(patient)

    # "Create New" Card
    next_idx = len(patients)
    with cols[next_idx % 3]:
        with st.container(border=True):
            st.markdown("<div style='height: 45px;'></div>", unsafe_allow_html=True)
            st.markdown("<p style='text-align: center; font-weight: bold; font-size: 1.2em;'>Nowy Pacjent</p>",
                        unsafe_allow_html=True)

            if st.button("Utwórz", key="btn_create_new", use_container_width=True, type="secondary"):
                open_create_modal()