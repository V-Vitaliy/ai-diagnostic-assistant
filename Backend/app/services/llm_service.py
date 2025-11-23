import os
import asyncio
import logging
import warnings
from google.genai import types
from google.adk.agents import Agent
from google.adk.runners import InMemoryRunner
from google.adk.tools import AgentTool, google_search

try:
    from prompt_builder import format_findings, format_patient_info
except ImportError:
    print("⚠️ Warning: prompt_builder.py not found. Using raw strings.")
    def format_findings(x): return str(x)
    def format_patient_info(x): return str(x)

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.ERROR)

MODEL_NAME = "gemini-2.0-flash"

try:
    if not os.environ.get("GOOGLE_API_KEY"):
        # os.environ["GOOGLE_API_KEY"] = "AIzaSy..."
        pass

    if not os.environ.get("GOOGLE_API_KEY"):
        raise ValueError("❌ GOOGLE_API_KEY environment variable is not set.")

    os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "FALSE"
    print("✅ Setup complete. API Key detected.")
except Exception as e:
    print(f"Authentication Error: {e}")
    exit()

# --- Badacz ---
def create_research_agent(model_name: str) -> Agent:
    return Agent(
        name="AgentBadacz",
        model=model_name,
        instruction="""
        Jesteś analitykiem medycznym (Medical Researcher). 
        Twoim jedynym zadaniem jest weryfikacja faktów (Fact-Checking) przy użyciu narzędzia `Google Search`.
        
        TWOJE ZADANIE:
        1. Otrzymasz objawy pacjenta i wstępne wyniki analizy AI.
        2. Wyszukaj w Google aktualne wytyczne medyczne i protokoły.
        3. Sprawdź korelacje: Czy te objawy są typowe dla wykrytej patologii?
        4. Sprawdź sprzeczności: Czy wyniki AI mogą być błędem (fałszywie dodatnie) w kontekście objawów?
        
        FORMAT WYJŚCIOWY (Zwróć tylko to):
        - WERYFIKACJA: [Potwierdzone / Sprzeczne / Niepewne]
        - ANALIZA RÓŻNICOWA: [Jakie inne choroby dają takie objawy?]
        - CZERWONE FLAGI: [Ostrzeżenia o zagrożeniu życia lub błędach AI]
        - ŹRÓDŁA: [Linki do znalezionych artykułów medycznych]
        """,
        tools=[google_search],
        output_key="research_findings",
    )

# --- Lekarz ---
def create_writer_agent(model_name: str) -> Agent:
    return Agent(
        name="AgentLekarz",
        model=model_name,
        instruction="""
        Jesteś doświadczonym lekarzem specjalistą.
        Twoim zadaniem jest napisanie końcowego raportu dla pacjenta i lekarza prowadzącego.
        
        DANE WEJŚCIOWE:
        - Przeczytaj wyniki badań dostarczone przez Agenta Badacza: {research_findings}
        - Uwzględnij kontekst pacjenta przekazany w rozmowie.
        
        INSTRUKCJE DOTYCZĄCE RAPORTU:
        1. Język: POLSKI. Ton: Profesjonalny, ale zrozumiały, empatyczny.
        2. Nie stawiaj ostatecznej diagnozy (jesteś AI), używaj fraz: "Obraz może sugerować...", "Wskazana konsultacja...".
        3. Jeśli Agent Badacz znalazł sprzeczności, wyraźnie zaznacz to w sekcji "UWAGI".
        
        STRUKTURA RAPORTU:
        ### 1. PODSUMOWANIE ANALIZY
        (Krótki opis tego, co wykryło AI i jak to się ma do objawów)
        
        ### 2. PRAWDOPODOBNE ROZPOZNANIE (Diagnostyka różnicowa)
        (Lista możliwych przyczyn uszeregowana od najbardziej prawdopodobnych)
        
        ### 3. REKOMENDACJE I DALSZE KROKI
        (Konkretne badania do wykonania: np. TK, morfologia, konsultacja kardiologiczna)
        """,
    )

# --- Orchestrator ---
async def run_medical_orchestrator(symptoms: str, findings: dict, patient_info: dict):
    print("\n🚀 Uruchamianie Systemu Diagnostycznego...")

    formatted_findings = format_findings(findings)
    formatted_patient = format_patient_info(patient_info)

    researcher = create_research_agent(MODEL_NAME)
    writer = create_writer_agent(MODEL_NAME)

    root_agent = Agent(
        name="KoordynatorMedyczny",
        model=MODEL_NAME,
        instruction="""
        Jesteś Głównym Koordynatorem Diagnostycznym. Zarządzasz procesem analizy przypadku medycznego.
        
        TWOJA PROCEDURA (Wykonaj krok po kroku):
        1. Wywołaj narzędzie `AgentBadacz`, aby zweryfikować spójność objawów z wynikami analizy AI. Przekaż mu dane pacjenta.
        2. Po otrzymaniu analizy od badacza, wywołaj narzędzie `AgentLekarz`, aby wygenerował końcowy raport PDF/Tekst.
        3. Jako swoją ostateczną odpowiedź zwróć TYLKO treść wygenerowaną przez `AgentLekarz`.
        """,
        tools=[
            AgentTool(researcher),
            AgentTool(writer)
        ]
    )

    user_task = f"""
    ANALIZA PRZYPADKU MEDYCZNEGO:
    
    --- DANE PACJENTA ---
    {formatted_patient}
    
    --- ZGŁOSZONE OBJAWY ---
    "{symptoms}"
    
    --- WYNIKI ANALIZY OBRAZOWEJ (AI) ---
    {formatted_findings}
    
    Rozpocznij procedurę badawczą (Krok 1: Weryfikacja, Krok 2: Raport).
    """

    print("🔄 Przetwarzanie danych...")

    runner = InMemoryRunner(agent=root_agent)
    response = await runner.run_debug(user_task)

    return response

async def main():
    test_symptoms = "Silny ból zamostkowy, promieniujący do lewej ręki, zimne poty."

    test_findings = {
        "Cardiomegaly": 0.85,
        "Pneumonia": 0.12,
        "heatmap_path": "/tmp/heat.png"
    }

    test_patient = {
        "age": 62,
        "gender": "Mężczyzna",
        "chronic_diseases": ["Nadciśnienie", "Cukrzyca typu 2"],
        "allergies": ["Penicylina"]
    }

    await run_medical_orchestrator(
        symptoms=test_symptoms,
        findings=test_findings,
        patient_info=test_patient
    )

if __name__ == "__main__":
    asyncio.run(main())