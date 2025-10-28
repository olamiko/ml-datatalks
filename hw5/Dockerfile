FROM agrigorev/zoomcamp-model:2025

WORKDIR /code

RUN pip install uv

COPY predict_service_v2.py predict_service.py

COPY pyproject.toml pyproject.toml

RUN uv sync

ENTRYPOINT ["uv",  "run",  "uvicorn", "predict_service:app", "--host",  "0.0.0.0", "--port", "9696"]
