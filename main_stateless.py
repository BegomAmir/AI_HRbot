# main_stateless.py — STATLESS endpoints (no server-side storage)
import os, json
from typing import Dict, Any, List, Optional
import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

RU_SYSTEM = (
    "Вы — HR-аватар компании. Общайтесь ТОЛЬКО на русском языке, на «вы», "
    "доброжелательно и профессионально. Никогда не переключайтесь на английский. "
    "В ответах для внутренних этапов возвращайте строго валидный JSON без лишнего текста."
)

client: Optional[httpx.AsyncClient] = None

async def chat_completion(messages: List[Dict[str, Any]],
                          model: Optional[str] = None,
                          base_url: Optional[str] = None,
                          api_key: Optional[str] = None,
                          temperature: float = 0.2,
                          max_tokens: int = 700) -> str:
    global client
    if client is None:
        raise RuntimeError("HTTP client is not initialized")
    model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
    base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
    api_key = api_key or os.getenv("LLM_API_KEY", "lm-studio")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json; charset=utf-8"}
    if api_key: headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        r = await client.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"LLM error: {e}")

def safe_json_loads(s: str) -> Dict[str, Any]:
    import re, json as _j
    cands=[s] if isinstance(s,str) else []
    if isinstance(s,str):
        cands += re.findall(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.I)
        if "{" in s and "}" in s:
            i,j=s.find("{"),s.rfind("}")
            if j>i>=0: cands.append(s[i:j+1])
    tried=set()
    for c in cands:
        if not isinstance(c,str): continue
        c=c.strip()
        if not c or c in tried: continue
        tried.add(c)
        try: return _j.loads(c)
        except Exception: pass
        try: return _j.loads(c.replace("'", '"'))
        except Exception: pass
    if os.getenv("DEBUG_LLM","0")=="1":
        with open("llm_last.txt","w",encoding="utf-8") as f: f.write(s if isinstance(s,str) else repr(s))
    raise HTTPException(status_code=502, detail="LLM returned non-JSON. Enable DEBUG_LLM=1 to dump raw output.")

RESUME_COMPRESS_PROMPT = """
Вы — ResumeCompressor. ТОЛЬКО русский. Верните ТОЛЬКО валидный JSON по схеме:
{"summary":"2–4 предложения","skills":[],"experience_years_by_skill":{},"notable_projects":[{"name":"","what":"","skills":[]}],"education":"","evidence_by_skill":{}}
"""
RESUME_CLAIMS_PROMPT = """
Вы — ClaimExtractor. На основе исходного текста резюме и его сжатой версии выделите ПРОВЕРЯЕМЫЕ утверждения (claims).
Каждый claim должен быть коротким, конкретным и с потенциальной верификацией в разговоре.
Верните ТОЛЬКО JSON:
{"claims":[{"id":"C1","text":"строил ETL в Airflow 2+ лет","skills":["Python","Airflow"],"kind":"experience|project|tool","criticality":"H|M|L"}]}
"""
QUESTION_PLANNER_PROMPT = """
Вы — InterviewPlanner. Создайте конкретные вежливые вопросы на русском.
Верните ТОЛЬКО JSON:
{"prioritized_questions":[{"id":"q1","skill":"строка","question":"вежливый вопрос","reason":"зачем","severity":"H|M|L","expected_signals":["..."]}]}
Сфокусируйтесь на навыках с наибольшими весами и на зонах неопределенности резюме.
"""
TURN_POLICY_PROMPT = """
Вы — InterviewPolicy. Работаете ТОЛЬКО на русском.
Вход: JD-веса, сжатое резюме, сводка разговора, последние реплики, текущие оценки по навыкам,
список claims и их текущие статусы, текущий вопрос и ответ.
Задача: понять, насколько ответ поддерживает/опровергает «claims», оценить консистентность, решить следующий ход.
Верните ТОЛЬКО JSON с полями как в предыдущей версии (agent_preface_ru, followup_question_ru, next_topic_question_ru, skill_scores, claim_updates, consistency_score, disqualify...). 
"""
FINAL_REPORT_PROMPT = """
Вы — InterviewReporter. Создайте отчёт на русском. Верните ТОЛЬКО JSON:
{"overall_score":0.0,"decision":"advance|reject|clarify","thresholds":{"advance":0.75,"clarify":0.6},"skills_breakdown":[{"skill":"Python","score":0.8,"weight":0.4,"evidence":[]}],"strengths":[],"gaps":[],"red_flags":[],"recommendation":"","candidate_feedback":[]}
"""

async def compress_resume_llm(resume_text: str, model: Optional[str]=None) -> Dict[str, Any]:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content":RESUME_COMPRESS_PROMPT+"\n\nРЕЗЮМЕ:\n"+resume_text}
    out=await chat_completion([system,user],model=model,temperature=0.1,max_tokens=800)
    return safe_json_loads(out)

async def extract_claims_llm(resume_text: str, resume_json: Dict[str, Any], model: Optional[str]=None) -> List[Dict[str, Any]]:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content":RESUME_CLAIMS_PROMPT+"\n\nИСХОДНОЕ_РЕЗЮМЕ:\n"+resume_text+"\n\nСЖАТОЕ_РЕЗЮМЕ:\n"+json.dumps(resume_json,ensure_ascii=False)}
    out=await chat_completion([system,user],model=model,temperature=0.1,max_tokens=600)
    return safe_json_loads(out).get("claims", [])

async def plan_questions_llm(jd_weights: Dict[str, float], resume_json: Dict[str, Any],
                             model: Optional[str]=None, num_questions:int=5) -> Dict[str, Any]:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content":QUESTION_PLANNER_PROMPT+
          f"\n\nСделайте минимум {num_questions} вопросов (лучше ровно {num_questions})."+
          f"\n\nJD_WEIGHTS:\n{json.dumps(jd_weights,ensure_ascii=False)}"+
          f"\n\nRESUME_JSON:\n{json.dumps(resume_json,ensure_ascii=False)}"}
    out=await chat_completion([system,user],model=model,temperature=0.1,max_tokens=700)
    return safe_json_loads(out)

async def rolling_summary_llm(prev_summary: str, role: str, text: str, model: Optional[str]=None) -> str:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content": "Вы — RollingSummarizer. Обновите сводку на русском (<120 слов). Верните ТОЛЬКО JSON: {\"rolling_summary\":\"...\"}\n\n"
                                   + json.dumps({"prev_summary":prev_summary,"new_turn":{"role":role,"text":text}},ensure_ascii=False)}
    out=await chat_completion([system,user],model=model,temperature=0.1,max_tokens=220)
    try: return safe_json_loads(out).get("rolling_summary",prev_summary)
    except Exception: return prev_summary

async def turn_policy_llm(payload: Dict[str,Any], model: Optional[str]=None) -> Dict[str,Any]:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content": TURN_POLICY_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False)}
    out=await chat_completion([system,user],model=model,temperature=0.1,max_tokens=900)
    return safe_json_loads(out)

async def final_report_llm(payload: Dict[str,Any], model: Optional[str]=None) -> Dict[str, Any]:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content": FINAL_REPORT_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False)}
    out=await chat_completion([system,user],model=model,temperature=0.1,max_tokens=900)
    return safe_json_loads(out)

from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="HR Avatar API (RU) — Stateless", version="1.0.1")

@app.on_event("startup")
async def _startup():
    global client
    client = httpx.AsyncClient()

@app.on_event("shutdown")
async def _shutdown():
    global client
    if client: await client.aclose()

class CompressReq(BaseModel):
    resume: str
    model: Optional[str] = None
    with_claims: bool = True

class CompressResp(BaseModel):
    resume_compressed: Dict[str, Any]
    claims: Optional[List[Dict[str, Any]]] = None

class InteractState(BaseModel):
    rolling_summary: str = ""
    recent_turns: List[Dict[str,str]] = []
    coverage_map: Dict[str, Dict[str, Any]] = {}
    claim_status: Dict[str,str] = {}
    consistency_score: float = 1.0
    consistency_events: List[Dict[str, Any]] = []
    evidence: Dict[str, List[str]] = {}
    question_queue: List[Dict[str, Any]] = []
    last_question: Optional[Dict[str, Any]] = None

class InteractReq(BaseModel):
    jd_weights: Dict[str, float]
    resume_compressed: Dict[str, Any]
    claims: List[Dict[str, Any]] = []
    model: Optional[str] = None
    state: InteractState = Field(default_factory=InteractState)
    candidate_text: Optional[str] = None
    num_questions: int = 5
    consistency_threshold: float = 0.45
    max_inconsistency: int = 2

class InteractResp(BaseModel):
    next_question: Optional[Dict[str, Any]] = None
    finished: bool = False
    reason: Optional[str] = None
    agent_preface_ru: Optional[str] = None
    comment_for_candidate_ru: Optional[str] = None
    state: InteractState

class FinalReq(BaseModel):
    jd_weights: Dict[str, float]
    resume_compressed: Dict[str, Any]
    evidence: Dict[str, List[str]] = {}
    coverage_map: Dict[str, Dict[str, Any]] = {}
    consistency_score: float = 1.0
    consistency_supported: List[Dict[str,Any]] = []
    consistency_refuted: List[Dict[str,Any]] = []
    model: Optional[str] = None

class FinalResp(BaseModel):
    report: Dict[str, Any]

@app.post("/compress_resume", response_model=CompressResp)
async def compress_resume_ep(req: CompressReq):
    rj = await compress_resume_llm(req.resume, model=req.model)
    claims = await extract_claims_llm(req.resume, rj, model=req.model) if req.with_claims else None
    return CompressResp(resume_compressed=rj, claims=claims)

@app.post("/interact_stateless", response_model=InteractResp)
async def interact_stateless_ep(req: InteractReq):
    st = req.state
    if not st.last_question:
        plan = await plan_questions_llm(req.jd_weights, req.resume_compressed, model=req.model, num_questions=req.num_questions)
        st.question_queue = [q for q in plan.get("prioritized_questions",[]) if q.get("question")]
        st.last_question = st.question_queue.pop(0) if st.question_queue else None
        return InteractResp(next_question=st.last_question, finished=False, reason=None, state=st)
    st.recent_turns.append({"role":"agent","text": st.last_question.get("question","")})
    st.rolling_summary = await rolling_summary_llm(st.rolling_summary, "agent", st.last_question.get("question",""), model=req.model)
    st.recent_turns.append({"role":"candidate","text": req.candidate_text or ""})
    st.rolling_summary = await rolling_summary_llm(st.rolling_summary, "candidate", req.candidate_text or "", model=req.model)
    payload = {
        "jd_weights": req.jd_weights,
        "resume_json": req.resume_compressed,
        "rolling_summary": st.rolling_summary,
        "recent_turns": st.recent_turns[-8:],
        "current_skill_scores": {k: st.coverage_map.get(k,{}).get("score",0.0) for k in req.jd_weights.keys()},
        "claims": req.claims,
        "claim_status": st.claim_status,
        "question_obj": st.last_question,
        "answer_text": req.candidate_text or ""
    }
    pol = await turn_policy_llm(payload, model=req.model)
    skill = st.last_question.get("skill","General")
    prev = st.coverage_map.get(skill, {"asked":0,"answered":0,"score":0.0})
    sc = float(pol.get("addressed_score", prev["score"]) or 0.0)
    st.coverage_map[skill] = {"asked":prev["asked"]+1,"answered":prev["answered"]+1,"score":sc}
    new_evs = pol.get("new_evidence")
    if isinstance(new_evs,str): new_evs=[new_evs]
    for ev in (new_evs or []): st.evidence.setdefault(skill,[]).append(ev)
    for upd in (pol.get("claim_updates") or []):
        cid = upd.get("id"); status = upd.get("status")
        if cid and status:
            st.claim_status[cid] = status
            st.consistency_events = (st.consistency_events or []) + [{"id": cid, "status": status, "evidence": upd.get("evidence","")}]
    cur_cons = float(pol.get("consistency_score", st.consistency_score) or 0.0)
    st.consistency_score = max(0.0, min(1.0, 0.5*st.consistency_score + 0.5*cur_cons))
    disq = bool(pol.get("disqualify", False))
    finished = disq or (st.consistency_score < req.consistency_threshold)
    reason = pol.get("disqualification_reason_ru") if finished else None
    next_q = None
    if not finished:
        na = pol.get("next_action","next_topic")
        if na == "ask_followup" and pol.get("followup_question_ru"):
            next_q = {"skill": skill, "question": pol["followup_question_ru"], "reason": "follow-up"}
        elif na == "next_topic":
            if pol.get("next_topic_question_ru"):
                next_q = {"skill": "(policy)", "question": pol["next_topic_question_ru"], "reason":"policy-suggested"}
            else:
                if not st.question_queue:
                    plan = await plan_questions_llm(req.jd_weights, req.resume_compressed, model=req.model, num_questions=req.num_questions)
                    st.question_queue = [q for q in plan.get("prioritized_questions",[]) if q.get("question")]
                if st.question_queue:
                    next_q = st.question_queue.pop(0)
        elif na == "end":
            finished = True
            reason = reason or "Спасибо, на этом всё."
        else:
            if st.question_queue: next_q = st.question_queue.pop(0)
    st.last_question = next_q
    return InteractResp(next_question=next_q, finished=finished, reason=reason,
                        agent_preface_ru=pol.get("agent_preface_ru",""),
                        comment_for_candidate_ru=pol.get("comment_for_candidate_ru",""),
                        state=st)

@app.post("/final_stateless", response_model=FinalResp)
async def final_stateless_ep(req: FinalReq):
    final_scores = {k: req.coverage_map.get(k,{}).get("score",0.0) for k in req.jd_weights.keys()}
    consistency = {"score": req.consistency_score,
                   "supported": req.consistency_supported[:20],
                   "refuted": req.consistency_refuted[:20]}
    payload = {"jd_weights": req.jd_weights, "resume_json": req.resume_compressed,
               "final_skill_scores": final_scores, "consistency": consistency, "evidence": req.evidence}
    report = await final_report_llm(payload, model=req.model)
    return FinalResp(report=report)

@app.get("/healthz")
async def healthz():
    ok=True
    try:
        _ = await chat_completion([{"role":"system","content":RU_SYSTEM},
                                   {"role":"user","content":"Ответьте числом: 1"}],
                                  temperature=0.0, max_tokens=5)
    except Exception:
        ok=False
    return {"status":"ok","llm_ok":ok}
