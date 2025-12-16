import asyncio
from datetime import date
from sqlalchemy import select
from app.db.session import AsyncSessionLocal, async_engine
from app.db.base import Base
from app.db.models.patient import Patient
from app.db.models.user import User
from app.core.security import get_password_hash

async def init_db():
    print("🔄 Creating database tables...")
    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    print("✅ Tables created successfully.")

    async with AsyncSessionLocal() as session:
        print("🌱 Starting database seed...")

        # 1. Create Test User (Doctor)
        result = await session.execute(select(User).where(User.email == "admin@medichat.com"))
        user = result.scalar_one_or_none()

        if not user:
            print("Creating test doctor account...")
            user = User(
                email="admin@medichat.com",
                hashed_password=get_password_hash("admin"),
                full_name="Dr. Adam Nowak"
            )
            session.add(user)
            await session.commit()
            await session.refresh(user)
            print(f"✅ Created user: {user.email} (Password: admin)")
        else:
            print(f"User {user.email} already exists. Skipping.")

        # 2. Create Test Patients linked to this doctor
        patients_data = [
            {
                "name": "Jan Kowalski",
                "gender": "M",
                "birth_date": date(1980, 5, 12),
                "chronic_diseases": ["Nadciśnienie", "Astma oskrzelowa"],
                "allergies": ["Orzechy", "Pyłki traw"],
                "medications": ["Aspirin 75mg", "Ventolin"],
                "height_cm": 182,
                "weight_kg": 88.5
            },
            {
                "name": "Anna Malinowska",
                "gender": "F",
                "birth_date": date(1995, 11, 23),
                "chronic_diseases": ["Hashimoto", "Insulinooporność"],
                "allergies": ["Penicylina"],
                "medications": ["Euthyrox N 50", "Glucophage XR"],
                "height_cm": 168,
                "weight_kg": 62.0
            },
            {
                "name": "Piotr Zieliński",
                "gender": "M",
                "birth_date": date(1965, 3, 8),
                "chronic_diseases": ["Cukrzyca typu 2", "Choroba wieńcowa"],
                "allergies": [],
                "medications": ["Metformina", "Bisoprolol", "Atorwastatyna"],
                "height_cm": 176,
                "weight_kg": 94.0
            }
        ]

        for p_data in patients_data:
            result = await session.execute(select(Patient).where(Patient.name == p_data["name"]))
            existing_patient = result.scalar_one_or_none()

            if not existing_patient:
                new_patient = Patient(
                    name=p_data["name"],
                    gender=p_data["gender"],
                    birth_date=p_data["birth_date"],
                    chronic_diseases=p_data["chronic_diseases"],
                    allergies=p_data["allergies"],
                    medications=p_data["medications"],
                    height_cm=p_data["height_cm"],
                    weight_kg=p_data["weight_kg"],
                    user_id=user.id  #
                )
                session.add(new_patient)
                print(f"✅ Created patient: {new_patient.name}")
            else:
                print(f"Patient {p_data['name']} already exists.")

        await session.commit()
        print("🚀 Database seeding completed successfully!")

if __name__ == "__main__":
    asyncio.run(init_db())