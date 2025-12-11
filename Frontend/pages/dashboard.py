import streamlit as st
from datetime import datetime

from app.api import fetch_patients, init_chat_session, create_patient


def inject_dashboard_css():
    """Unified CSS styling for professional look"""
    st.markdown("""
    <style>
        /* Global styling */
        .main { background-color: #0E1117; }

        /* Patient cards - compact unified design */
        .patient-card {
            background: linear-gradient(145deg, #1a1d29 0%, #151820 100%);
            border: 1px solid #2d3139;
            border-radius: 16px;
            padding: 18px;
            transition: all 0.3s ease;
            box-shadow: 0 4px 12px rgba(0,0,0,0.3);
            min-height: 180px;
            display: flex;
            flex-direction: column;
        }
        .patient-card:hover {
            border-color: #4e4376;
            box-shadow: 0 8px 24px rgba(78, 67, 118, 0.2);
            transform: translateY(-2px);
        }

        /* Patient header with avatar */
        .patient-header-row {
            display: flex;
            align-items: center;
            gap: 12px;
            margin-bottom: 14px;
        }

        .patient-avatar {
            width: 48px;
            height: 48px;
            border-radius: 50%;
            background: linear-gradient(135deg, #2b5876 0%, #4e4376 100%);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1.3rem;
            font-weight: 600;
            color: white;
            flex-shrink: 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.3);
        }

        .patient-name-block {
            flex: 1;
            min-width: 0;
        }

        .patient-name {
            font-size: 1.1rem;
            font-weight: 700;
            color: #ffffff;
            letter-spacing: 0.3px;
            margin: 0;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .patient-age {
            color: #7c8db5;
            font-size: 0.85rem;
            margin-top: 2px;
        }

        /* Medical info grid */
        .medical-info-grid {
            display: grid;
            gap: 8px;
            margin-bottom: 12px;
            flex: 1;
        }

        .medical-item {
            background-color: rgba(45, 49, 57, 0.4);
            padding: 8px 10px;
            border-radius: 8px;
            border-left: 3px solid #4e4376;
        }

        .medical-label {
            color: #7c8db5;
            font-size: 0.75rem;
            font-weight: 600;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 3px;
        }

        .medical-value {
            color: #c5c9d4;
            font-size: 0.85rem;
            line-height: 1.3;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }

        .no-data-text {
            color: #6b7280;
            font-size: 0.85rem;
            font-style: italic;
            text-align: center;
            padding: 20px 0;
        }

        /* New patient card */
        .new-patient-card {
            background: linear-gradient(145deg, #1e2432 0%, #181b25 100%);
            border: 2px dashed #3d4352;
            border-radius: 16px;
            padding: 20px;
            text-align: center;
            transition: all 0.3s ease;
            min-height: 180px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
        }
        .new-patient-card:hover {
            border-color: #5a6073;
            background: linear-gradient(145deg, #232838 0%, #1d2028 100%);
        }

        .new-patient-icon {
            font-size: 2.2rem;
            margin-bottom: 8px;
            opacity: 0.7;
        }

        .new-patient-text {
            color: #9ba3b8;
            font-size: 1rem;
            font-weight: 600;
            letter-spacing: 0.3px;
        }

        /* Modal styling */
        .modal-header {
            color: #7c8db5;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 0.5px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)


def calculate_age(birth_date_str):
    """Calculates age based on date of birth"""
    if not birth_date_str: return "N/A"
    try:
        birth = datetime.strptime(birth_date_str, "%Y-%m-%d")
        today = datetime.today()
        return today.year - birth.year - ((today.month, today.day) < (birth.month, birth.day))
    except:
        return "?"


def get_initials(name):
    """Get initials from name for avatar"""
    if not name:
        return "?"
    parts = name.strip().split()
    if len(parts) >= 2:
        return f"{parts[0][0]}{parts[1][0]}".upper()
    return parts[0][0].upper() if parts else "?"


def handle_click(patient):
    """Handler for patient selection: starts session and redirects to chat"""
    with st.spinner(f"Ładowanie karty pacjenta {patient['name']}..."):
        session_data = init_chat_session(patient['id'])

        if session_data and 'session_id' in session_data:
            st.session_state['session_id'] = session_data['session_id']
            history = session_data.get('history_json', [])
            st.session_state['chat_history'] = history
            st.session_state['current_patient'] = patient
            st.session_state['patient_data'] = patient
            st.session_state['page'] = 'chat'
            st.rerun()
        else:
            st.error("Błąd serwera: nie udało się utworzyć sesji lub pobrać historii.")


@st.dialog("Nowy pacjent")
def open_create_modal():
    st.markdown('<div class="modal-header">Wypełnij dane, aby utworzyć nową kartę pacjenta</div>',
                unsafe_allow_html=True)

    with st.form("create_patient_form"):
        name = st.text_input("Imię i Nazwisko", placeholder="Jan Kowalski")
        birth_date = st.date_input("Data urodzenia", min_value=datetime(1900, 1, 1), value=datetime(1990, 1, 1))

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
                    "weight_kg": int(weight)
                }

                if create_patient(new_patient_data):
                    st.success("Pacjent został pomyślnie utworzony!")
                    st.rerun()
                else:
                    st.error("Nie udało się zapisać pacjenta. Sprawdź logi.")


def render():
    inject_dashboard_css()

    st.title("🏥 Panel Pacjentów")
    st.markdown("---")

    patients = fetch_patients()
    cols = st.columns(3)

    # Loop through existing patients
    for idx, patient in enumerate(patients):
        with cols[idx % 3]:
            patient_name = patient.get('name', 'Bez nazwy')
            age = calculate_age(patient.get('birth_date'))
            initials = get_initials(patient_name)

            # Build medical info
            chronic = patient.get('chronic_diseases', [])
            allergies = patient.get('allergies', [])
            medications = patient.get('medications', [])

            medical_items = []
            if chronic and len(chronic) > 0:
                chronic_text = ", ".join(chronic)
                medical_items.append(
                    f'<div class="medical-item"><div class="medical-label">Choroby</div><div class="medical-value">{chronic_text}</div></div>')

            if allergies and len(allergies) > 0:
                allergies_text = ", ".join(allergies)
                medical_items.append(
                    f'<div class="medical-item"><div class="medical-label">Alergie</div><div class="medical-value">{allergies_text}</div></div>')

            if medications and len(medications) > 0:
                meds_text = ", ".join(medications)
                medical_items.append(
                    f'<div class="medical-item"><div class="medical-label">Leki</div><div class="medical-value">{meds_text}</div></div>')

            medical_html = "".join(
                medical_items) if medical_items else '<div class="no-data-text">Brak danych medycznych</div>'

            # Render card
            st.markdown(f'''
                <div class="patient-card">
                    <div class="patient-header-row">
                        <div class="patient-avatar">{initials}</div>
                        <div class="patient-name-block">
                            <div class="patient-name">{patient_name}</div>
                            <div class="patient-age">{age} lat</div>
                        </div>
                    </div>
                    <div class="medical-info-grid">
                        {medical_html}
                    </div>
                </div>
            ''', unsafe_allow_html=True)

            # Button
            st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
            if st.button("💬 Przejdź do czatu", key=f"btn_{patient['id']}", use_container_width=True, type="primary"):
                handle_click(patient)

    # "Create New" Card
    next_idx = len(patients)
    with cols[next_idx % 3]:
        st.markdown('''
            <div class="new-patient-card">
                <div class="new-patient-icon">➕</div>
                <div class="new-patient-text">Nowy Pacjent</div>
            </div>
        ''', unsafe_allow_html=True)

        st.markdown("<div style='margin-top: 10px;'></div>", unsafe_allow_html=True)
        if st.button("Utwórz profil", key="btn_create_new", use_container_width=True, type="secondary"):
            open_create_modal()