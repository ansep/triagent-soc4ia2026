FROM python:3.11-slim

WORKDIR /app

# CPU-only torch: the runner uses it only for the sentence-transformers embeddings,
# and the CUDA wheel would add several GB to the image
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ .
COPY experiments/ /app/experiments/

ENV PYTHONUNBUFFERED=1
# runs the CLI (batch / monolithic / seed); for the FastAPI server:
#   docker run ... --entrypoint uvicorn triagent-runner main:app --host 0.0.0.0
ENTRYPOINT ["python", "-m", "cli"]
