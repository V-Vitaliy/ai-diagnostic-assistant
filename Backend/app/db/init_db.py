import asyncio
from datetime import date
from sqlalchemy import select
from app.db.session import AsyncSessionLocal
from app.db.models import Patient, AnalysisResult, ChatSession

async def init_db():
    async with AsyncSessionLocal() as session:
        result = await session.execute(select(Patient))
        patient = result.scalars().first()

        if not patient:
            print("Creating test patient...")
            new_patient = Patient(
                name="Jan Kowalski",
                birth_date=date(1980, 5, 12),
                chronic_diseases=["Nadciśnienie", "Astma"],
                allergies=["Orzechy", "Pyłki"],
                medications=["Aspirin", "Ventolin"]
            )
            session.add(new_patient)
            await session.commit()
            print(f"✅ Created patient: {new_patient.name} (ID: {new_patient.id})")
        else:
            print("Skipping seed: Data already exists.")

if __name__ == "__main__":
    asyncio.run(init_db())