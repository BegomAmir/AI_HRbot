## Требования
- Python 3.11+ и `pip`, либо Docker.
- Запущенный LLM‑сервер (OpenAI‑совместимый, например LM Studio) на `http://127.0.0.1:1234`.
- Windows PowerShell: при проблемах кодировки используйте `chcp 65001` и отправляйте JSON как UTF‑8 байты.

## Установка и запуск (локально)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
$env:LLM_BASE_URL="http://127.0.0.1:1234"; $env:LLM_MODEL="openai/gpt-oss-20b"
uvicorn main_stateless:app --host 0.0.0.0 --port 8000 --reload
```

## Запуск в Docker
```powershell
docker build -t hr-ml-ru-stateless:v4 .
docker run --rm -p 8000:8000 -e LLM_BASE_URL=http://host.docker.internal:1234 -e LLM_MODEL=openai/gpt-oss-20b hr-ml-ru-stateless:v4
# или
docker compose up --build
```

## Эндпоинты
- `POST /compress_resume` → `{ resume: string, with_claims?: bool, model?: string }` ⇒ `{ resume_compressed: {...}, claims?: [...] }`
- `POST /interact_stateless`
  - Первый вызов (без `state.last_question`) возвращает план: `{ next_question, state }`
  - Далее: передаём `candidate_text` и **весь state** из прошлого ответа ⇒ новый вопрос и обновлённый `state`
- `POST /final_stateless` → `{ jd_weights, resume_compressed, evidence, coverage_map, consistency_score }` ⇒ `{ report }`
- `GET /healthz` → `{ status, llm_ok }`

## Быстрая проверка (PowerShell, по одной строке)
```powershell
$api='http://localhost:8000'
```
```powershell
Invoke-RestMethod "$api/healthz" | ConvertTo-Json -Depth 10
```
```powershell
$body=@{resume='Python разработчик 3+ года; Airflow 2.x; PostgreSQL; asyncio; Docker';with_claims=$true}|ConvertTo-Json -Depth 20; $compress=Invoke-RestMethod -Uri "$api/compress_resume" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($body))
```
```powershell
$jd=@{Python=0.5;SQL=0.3;Airflow=0.2}; $planReq=@{jd_weights=$jd;resume_compressed=$compress.resume_compressed;claims=$compress.claims;num_questions=5}|ConvertTo-Json -Depth 30; $plan=Invoke-RestMethod -Uri "$api/interact_stateless" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($planReq))
```
```powershell
$ans='Использовал asyncio.gather и тайм-ауты'; $evalReq=@{jd_weights=$jd;resume_compressed=$compress.resume_compressed;claims=$compress.claims;candidate_text=$ans;state=$plan.state}|ConvertTo-Json -Depth 40; $eval=Invoke-RestMethod -Uri "$api/interact_stateless" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($evalReq))
```
```powershell
$finalReq=@{jd_weights=$jd;resume_compressed=$compress.resume_compressed;evidence=$eval.state.evidence;coverage_map=$eval.state.coverage_map;consistency_score=$eval.state.consistency_score}|ConvertTo-Json -Depth 40; $final=Invoke-RestMethod -Uri "$api/final_stateless" -Method Post -ContentType "application/json; charset=utf-8" -Body ([Text.Encoding]::UTF8.GetBytes($finalReq))
```

## Переменные окружения
- `LLM_BASE_URL` — базовый URL LLM (по умолчанию `http://127.0.0.1:1234` локально / `host.docker.internal:1234` в Docker)
- `LLM_MODEL` — идентификатор модели (например, `openai/gpt-oss-20b`)
- `LLM_API_KEY` — если требуется авторизация
