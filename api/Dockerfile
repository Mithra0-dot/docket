FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY main.py load_data.py train_model.py explain_model.py embed_cases.py investigate_cases.py risk_overview.py ./
RUN mkdir -p /app/models

EXPOSE 8000
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
