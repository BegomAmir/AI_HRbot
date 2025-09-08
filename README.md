# HR Avatar MVP v2 — hot context + backend hooks
Быстрый старт (Windows + LM Studio):
1) pip install requests
2) set env:
   $env:LLM_BASE_URL="http://127.0.0.1:1234"
   $env:LLM_API_KEY="lm-studio"
   $env:LLM_MODEL="openai/gpt-oss-20b"
3) Тест:
   python app.py --resume sample_resume.txt --jd sample_jd.txt --compress-only
4) Интервью:
   python app.py --resume sample_resume.txt --jd sample_jd.txt --max-turns 3
