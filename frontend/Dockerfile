FROM python:3.11-slim

WORKDIR /app

# Install only what the UI needs
RUN pip install streamlit requests

COPY ui.py .

EXPOSE 8501

CMD ["streamlit", "run", "ui.py", "--server.address=0.0.0.0"]