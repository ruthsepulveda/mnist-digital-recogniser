FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies required for opencv
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install --upgrade pip && pip install poetry

# Copy dependency files
COPY pyproject.toml poetry.lock* ./

# Install dependencies without virtual environment
RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

# Copy code and model
COPY main.py ./
COPY model.keras ./

# Expose Streamlit port
EXPOSE 7860

# Healthcheck to verify the app is running
HEALTHCHECK CMD curl -f http://localhost:7860/ || exit 1

# Entry command
CMD ["poetry", "run", "streamlit", "run", "main.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0"]

