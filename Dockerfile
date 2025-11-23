# 1. Base Image: Use a lightweight Python Linux image
FROM python:3.11-slim

# 2. Set working directory inside the container
WORKDIR /app

# 3. Install system dependencies (needed for some PDF tools)
RUN apt-get update && apt-get install -y build-essential libsqlite3-dev && rm -rf /var/lib/apt/lists/*

# 4. Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --default-timeout=1000 --no-cache-dir -r requirements.txt
# 5. Copy the source code
COPY src/ src/

# 6. Create the storage directory for ChromaDB
RUN mkdir -p db_storage

# 7. Expose the port FastAPI runs on
EXPOSE 8000

# 8. Command to run the app
# "0.0.0.0" to allow external access to the container
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]