## Установка

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### Переменные окружения
```bash
export LLM_BASE_URL="http://127.0.0.1:1234"
export LLM_API_KEY="lm-studio"
export LLM_MODEL="openai/gpt-oss-20b"
# опционально:
export DEBUG_LLM="1"           # дамп сырого ответа в llm_last.txt
export DRF_TURN_URL=""         # куда POST'ить финальные реплики
export DRF_API_KEY=""
export FRONT_SUBS_URL=""       # live-субтитры
export TTS_URL=""              # озвучка
```

## Запуск
```bash
# Только компрессия резюме + claims
python app_ru_dynamic_guard.py --resume sample_resume.txt --jd sample_jd.txt --compress-only

# Диалог (динамика + раннее завершение при несоответствиях)
python app_ru_dynamic_guard.py --resume sample_resume.txt --jd sample_jd.txt --max-turns 6 --min-turns 2
```

### Полезные флаги
- `--max-turns` — максимум **главных** вопросов (follow-up внутри не считается новым turn).
- `--min-turns` — минимум главных вопросов до завершения.
- `--consistency-threshold` — порог достоверности (0..1) для раннего завершения.
- `--max-inconsistency` — сколько критичных claims можно опровергнуть до отказа.
- `--min-probe` — минимум главных вопросов до «жёсткого» отказа.
- `--weights "Python 0.4, SQL 0.3, Коммуникация 0.3"` — вручную задать веса; иначе LLM берёт из JD.

## Выходные артефакты (`runs/run_<timestamp>/`)
- **report.json** — машинно-читаемый отчёт: `overall_score`, `decision`, `skills_breakdown`, `red_flags`, `candidate_feedback`, блок `consistency` (подтверждённые/опровергнутые утверждения и итоговый `score`).
- **report.md** — человеко-читаемое резюме отчёта.
- **evidence.json** — собранные подтверждения по навыкам (evidence из ответов).
- **coverage_map.json** — покрытие компетенций: сколько раз спрашивали/отвечали и текущий `score` по каждому навыку.
- **skill_scores.json** — финальные оценки навыков (0..1).
- **claims.json** — извлечённые из резюме проверяемые утверждения (начальный набор для проверки; актуальные статусы см. в `report.json → consistency`).
- **resume_compressed.json** — сжатое резюме (summary, skills, experience_by_skill, notable_projects, education, evidence_by_skill).
- **jd_weights.json** — веса компетенций (сумма = 1.0).
