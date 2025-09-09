## Установка и запуск
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:LLM_BASE_URL="http://127.0.0.1:1234"; $env:LLM_MODEL="openai/gpt-oss-20b"
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Docker
```powershell
docker build -t hr-ml-ru-stateless-oop .
docker run --rm -p 8000:8000 -e LLM_BASE_URL=http://host.docker.internal:1234 -e LLM_MODEL=openai/gpt-oss-20b hr-ml-ru-stateless-oop
```

## Эндпоинты
- POST /compress_resume
- POST /interact_stateless
- POST /final_stateless
- GET /healthz
