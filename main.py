import httpx
from fastapi import FastAPI
from app.api.routes import router, Container
from app.clients.llm import LLMClient
from app.services.compress import ResumeService
from app.services.interview import InterviewService
from app.services.report import ReportService
from app.core.prompts import RU_SYSTEM

app = FastAPI(title="HR Avatar API (RU) — Stateless OOP", version="2.0.1")
app.include_router(router)

@app.on_event("startup")
async def _startup():
    client = httpx.AsyncClient()
    llm = LLMClient(client=client)
    resume = ResumeService(llm)
    interview = InterviewService(llm)
    report = ReportService(llm)
    app.state.client = client
    app.state.llm = llm
    app.state.container = Container(resume=resume, interview=interview, report=report)

@app.on_event("shutdown")
async def _shutdown():
    client: httpx.AsyncClient = app.state.client
    await client.aclose()

@app.get("/healthz")
async def healthz():
    ok=True
    try:
        _ = await app.state.llm.chat_completion(
            [{"role":"system","content":RU_SYSTEM},{"role":"user","content":"Ответьте числом: 1"}],
            temperature=0.0, max_tokens=5
        )
    except Exception:
        ok=False
    return {"status":"ok","llm_ok":ok}
