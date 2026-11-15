# Heliograph on Hugging Face Spaces (Docker SDK, CPU).
# One container: FastAPI serves the API and the built React UI on port 7860.

# ---- build the UI ----------------------------------------------------------
FROM node:22-slim AS web
WORKDIR /src/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# ---- runtime ---------------------------------------------------------------
FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    HELIOGRAPH_SERVE_FRONTEND=1 HELIOGRAPH_DEVICE=cpu
RUN useradd -m -u 1000 user
WORKDIR /home/user/app

COPY backend/requirements.txt backend/requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r backend/requirements.txt

COPY --chown=user backend/ backend/
COPY --chown=user scripts/ scripts/
COPY --chown=user reference/ reference/
COPY --chown=user docs/ docs/
COPY --chown=user --from=web /src/frontend/dist frontend/dist
RUN mkdir -p data artifacts && chown -R user data artifacts

USER user
# Download and verify the NASA SMAP/MSL benchmark at build time (Kaggle, then the pinned GitHub copy).
RUN python scripts/acquire_data.py && python scripts/init_db.py

EXPOSE 7860
WORKDIR /home/user/app/backend
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
