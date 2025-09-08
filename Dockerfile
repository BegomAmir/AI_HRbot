FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
# LM Studio defaults (use host.docker.internal under Docker)
ENV LLM_BASE_URL=http://host.docker.internal:1234         LLM_API_KEY=lm-studio         LLM_MODEL=openai/gpt-oss-20b
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY main_stateless.py ./
EXPOSE 8000
CMD ["uvicorn", "main_stateless:app", "--host", "0.0.0.0", "--port", "8000"]
