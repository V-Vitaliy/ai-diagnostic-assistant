# AI Diagnostic Assistant

## 1. Opis Projektu

**AI Diagnostic Assistant** to zaawansowany system wspomagania decyzji klinicznych (CDSS), który łączy najnowocześniejsze osiągnięcia analizy obrazowej (Computer Vision) z przetwarzaniem języka naturalnego (LLM). System dostarcza lekarzom tzw. "drugą opinię", analizując zdjęcia RTG/TK pod kątem patologii i generując wyjaśnialne mapy aktywacji (XAI).

**Główne Funkcjonalności**:

- **Bezpieczna Autoryzacja**: System zarządzania dostępem oparty na protokole JWT (python-jose, passlib). Zapewnia pełną izolację danych pacjentów i bezpieczne logowanie.  
- **Analysis Workspace**: Moduł analizy obrazów medycznych (np. RTG klatki piersiowej) wykrywający patologie takie jak zapalenie płuc czy złamania.
- **Wyjaśnialna Sztuczna Inteligencja (XAI)**: Generowanie map ciepła przy użyciu algorytmu Grad-CAM, co pozwala lekarzowi zrozumieć, na których fragmentach obrazu model oparł swoją diagnozę.
- **Inteligentny Chatbot**: Interfejs konwersacyjny zintegrowany z Google ADK, umożliwiający zadawanie pytań dotyczących wyników badań oraz historii pacjenta.
- **Obsługa OCR**: Wykorzystanie EasyOCR do digitalizacji danych z dokumentacji medycznej.
 

## 2. Stos Technologiczny (Tech Stack)

| Komponent | Technologia | Rola w projekcie |
|----------|-------------|-------------|
| Frontend | Streamlit, Requests | Interaktywny interfejs użytkownika (UI/UX) i komunikacja z API. |
| Backend API | FastAPI, Uvicorn, Pydantic | Asynchroniczna logika biznesowa, routing i walidacja danych. |
| AI - Computer Vision | PyTorch, torchxrayvision, MONAI | Silniki detekcji, segmentacji i klasyfikacji obrazów medycznych. | 
| LLM / Chatbot | google-genai | Generowanie raportów medycznych i inteligentna konwersacja. |
| Baza Danych | PostgreSQL, SQLAlchemy, asyncpg | Relacyjne, asynchroniczne przechowywanie danych i wyników badań. |
| Migracje DB | Alembic | Wersjonowanie schematu bazy danych. |
| Konteneryzacja |Docker, Docker Compose | |
| Przetwarzanie Danych | pandas, scikit-learn, nibabel | Manipulacja danymi tabelarycznymi i obsługa formatów medycznych (NIfTI). |


## 3. Architektura i Przepływ Danych (Workflow)

System został zaprojektowany zgodnie z paradygmatem mikrousług, co zapewnia skalowalność i łatwość wdrażania nowych modeli.

**Warstwa Prezentacji** (Streamlit): Lekarz przesyła obraz diagnostyczny lub wprowadza objawy. Frontend komunikuje się z backendem za pomocą biblioteki requests.

**Warstwa Logiki** (FastAPI): Żądanie trafia do endpointu, gdzie następuje autoryzacja tokenem JWT. Dane są wstępnie przetwarzane i kolejkowane do silnika AI.

**Warstwa Inferencyjna** (PyTorch/XAI):

- Model klasyfikuje obraz (np. przy użyciu wag z torchxrayvision).

- Biblioteka pytorch-gradcam generuje wizualizację istotnych obszarów (Heatmaps).

- W przypadku dokumentów tekstowych, EasyOCR wyodrębnia kluczowe informacje do bazy wiedzy.

**Warstwa Persystencji** (PostgreSQL): Wszystkie wyniki analizy, metadane obrazu oraz historia sesji czatu są zapisywane asynchronicznie w bazie danych.
 Wykorzystane Modele AI i Standardy

## 4. Wykorzystane Modele AI i Standardy
W systemie zaimplementowano uznane architektury sieci neuronowych:

**DenseNet (CheXNet/MURA)**: Wykorzystywane do precyzyjnej klasyfikacji zdjęć RTG klatki piersiowej oraz badań układu kostno-szkieletowego.

**MONAI/wholeBody_ct_segmentation** (https://zenodo.org/record/6802614#.Y9iTydLMJ6I): Wykorzystanie standardów MONAI (Zenodo) do zaawansowanej segmentacji całego ciała w badaniach TK.

**TorchXrayVision**: Wykorzystanie modeli SOTA (State-of-the-art) wytrenowanych na ogromnych zbiorach danych radiologicznych.

**LightGBM**: Moduł wspierający analizę danych tabelarycznych (np. predykcja ryzyka na podstawie parametrów krwi).
