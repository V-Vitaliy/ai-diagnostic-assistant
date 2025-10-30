FROM python:3.10-slim

RUN apt-get update && apt-get install -y build-essential cmake

WORKDIR /app

# 5. Copy the AI model files into the app
# First, copy the custom model definition
COPY ./app/services/models/CustomModel.py /app/app/services/models/
# Second, copy the large model weights file
COPY ./app/services/models/model.pt /app/app/services/models/

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload"]
