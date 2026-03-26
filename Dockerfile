FROM python:3.10-slim

WORKDIR /usr/src/app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV GROQ_API_KEY=your_key_here

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]