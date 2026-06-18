FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy full project (including model folder)
COPY . .

# ensure model folder exists (safe fallback)
RUN mkdir -p model

EXPOSE 8080

CMD ["python", "app/app.py"]
