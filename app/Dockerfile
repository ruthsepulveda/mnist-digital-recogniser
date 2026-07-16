FROM python:3.12-bullseye

WORKDIR /app

RUN pip install --upgrade pip && pip install poetry

COPY app/pyproject.toml app/poetry.lock* ./

RUN poetry config virtualenvs.create false && \
    poetry install --no-interaction --no-ansi --no-root

COPY app/main.py ./
COPY app/model.keras ./

EXPOSE 7860

HEALTHCHECK CMD curl -f http://localhost:7860/ || exit 1

CMD ["poetry", "run", "streamlit", "run", "main.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0"]


