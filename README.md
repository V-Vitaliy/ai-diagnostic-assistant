# AI Diagnostic Assistant

## 1. Opis Projektu

**AI Diagnostic Assistant** to inteligentny system wsparcia diagnostycznego opracowany w ramach kursu *Projekt interdyscyplinarny*.  
Aplikacja wykorzystuje architekturę mikrousług (Docker) oraz modele sztucznej inteligencji do analizy obrazów medycznych i wspierania lekarzy w procesie decyzyjnym.

Główne funkcje systemu:

- Analiza obrazów medycznych (RTG klatki piersiowej, RTG kończyn)  
- Wykrywanie potencjalnych patologii (np. zapalenie płuc, złamania)  
- Generowanie map ciepła (Grad-CAM), wskazujących obszary kluczowe dla modelu  
- **W przyszłości:** integracja danych pacjenta oraz generowanie raportów tekstowych przy użyciu LLM  

## 2. Stos Technologiczny (Tech Stack)

| Komponent | Technologia |
|----------|-------------|
| Backend API | Python, FastAPI |
| Frontend UI | Python, Streamlit |
| Konteneryzacja | Docker, Docker Compose |
| Baza danych | PostgreSQL |
| AI – Obrazowanie | PyTorch, torchxrayvision, pytorch-grad-cam |
| Modele AI | DenseNet (CheXNet), własny DenseNet (MURA) |

## 3. Architektura Systemu

System składa się z trzech kontenerów Docker komunikujących się przez sieć wewnętrzną:

- **Frontend (Streamlit)** – interfejs użytkownika dostępny w przeglądarce; umożliwia przesyłanie obrazów i danych  
- **Backend (FastAPI)** – główny moduł logiczny; odbiera żądania, integruje modele AI, zarządza przepływem danych  
- **Database (PostgreSQL)** – przechowuje profile pacjentów oraz wyniki analiz  

## 4. Jak Uruchomić Projekt (Lokalnie)

Projekt jest w pełni skonteneryzowany — nie wymaga ręcznej instalacji bibliotek.

### Wymagania

- Docker Desktop  
- Model z wagami do detekcji złamań (model.pt) umieszczony w katalogu:  
  `backend/app/services/models/`

### Uruchomienie
1. Sklonuj repozytorium:
  git clone [URL-TWOJEGO-REPOZYTORIUM]
  cd ai-diagnostic-assistant
2. Uruchom platformę:
  docker-compose up --build
3. Otwórz aplikację:

- Frontend (UI): http://localhost:8501  
- Backend (API Docs – Swagger): http://localhost:8080/docs  

## 5. Zespół (Role)

| Rola | Odpowiedzialność |
|------|-------------------|
| Product Owner, Team Lead, AI/Backend Lead | Strategia, zarządzanie, architektura, modele AI |
| Backend Developer | API, baza danych, integracja usług |
| Frontend Developer (Lead) | Projektowanie i implementacja UI |
| Frontend Developer | Komponenty interfejsu |
| QA & Documentation Specialist | Testy, dokumentacja, zgodność z wymaganiami kursu |1. Sklonuj repozytorium:

