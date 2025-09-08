#!/usr/bin/env python3
import os, json, time, uuid, argparse, requests
from collections import deque
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# =============================
# RU-only system style
# =============================
RU_SYSTEM = (
    "Вы — HR-аватар компании. Общайтесь ТОЛЬКО на русском языке, на «вы», "
    "доброжелательно и профессионально. Никогда не переключайтесь на английский. "
    "В ответах для внутренних этапов возвращайте строго валидный JSON без лишнего текста."
)

# =============================
# OpenAI-compatible client
# =============================
def chat_completion(messages: List[Dict[str, Any]],
                    model: Optional[str] = None,
                    base_url: Optional[str] = None,
                    api_key: Optional[str] = None,
                    temperature: float = 0.2,
                    max_tokens: int = 700):
    model = model or os.getenv("LLM_MODEL", "gpt-4o-mini")
    base_url = base_url or os.getenv("LLM_BASE_URL", "http://127.0.0.1:1234")
    api_key = api_key or os.getenv("LLM_API_KEY", "lm-studio")
    url = base_url.rstrip("/") + "/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]

# =============================
# Robust JSON extraction
# =============================
def safe_json_loads(s: str):
    import re, json as _j
    cands = []
    if isinstance(s, str):
        cands.append(s)
        for m in re.findall(r"```(?:json)?\s*([\s\S]*?)```", s, flags=re.I):
            cands.append(m)
        if "{" in s and "}" in s:
            i, j = s.find("{"), s.rfind("}")
            if j > i >= 0: cands.append(s[i:j+1])
    tried=set()
    for c in cands:
        if not isinstance(c, str): continue
        c=c.strip()
        if not c or c in tried: continue
        tried.add(c)
        try: return _j.loads(c)
        except Exception: pass
        try: return _j.loads(c.replace("'", '"'))
        except Exception: pass
    if os.getenv("DEBUG_LLM","0")=="1":
        with open("llm_last.txt","w",encoding="utf-8") as f: f.write(s if isinstance(s,str) else repr(s))
    raise ValueError("LLM did not return valid JSON; set DEBUG_LLM=1 to dump raw output.")

# =============================
# Prompts (RU)
# =============================
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
Вход: JD-веса, сжатое резюме, исходное резюме, сводка разговора, последние реплики, текущие оценки по навыкам,
список claims (утверждений резюме) и их текущие статусы проверки, текущий вопрос и ответ.
Задача: понять, насколько ответ поддерживает/опровергает «claims», оценить консистентность, решить следующий ход.
Правила:
- Тон доброжелательный и уважительный.
- Если ответы системно НЕ подтверждают ключевые claims (высокая критичность) и накапливаются противоречия — рекомендуйте завершение диалога (disqualify).
- Перед финальным завершением можно предложить мягкую проверку/уточнение (consistency_probe_ru).

Верните ТОЛЬКО JSON:
{
  "signals": ["short_answer","uncertain","off_topic"],
  "addressed_skill": "строка",
  "addressed_score": 0.0,
  "new_evidence": [],
  "skill_scores": {},
  "red_flags": [],
  "contradictions": [],
  "claim_updates": [{"id":"C1","status":"supported|refuted|unclear","evidence":"кратко"}],
  "consistency_score": 0.0,            // 0..1 общая оценка достоверности резюме на текущий момент
  "disqualify": false,
  "disqualification_reason_ru": "",
  "path": "simplify|probe_deeper|shift_topic|clarify",
  "next_action": "ask_followup|next_topic|end",
  "followup_question_ru": null,
  "next_topic_question_ru": null,
  "consistency_probe_ru": null,        // 1 мягкий проверочный вопрос, если уместно
  "agent_preface_ru": "вежливая фраза",
  "comment_for_candidate_ru": "короткая поддержка/направление",
  "rationale": "кратко"
}
"""

FINAL_REPORT_PROMPT = """
Вы — InterviewReporter. Создайте отчёт на русском. Верните ТОЛЬКО JSON:
{"overall_score":0.0,"decision":"advance|reject|clarify","thresholds":{"advance":0.75,"clarify":0.6},"skills_breakdown":[{"skill":"Python","score":0.8,"weight":0.4,"evidence":[]}],"strengths":[],"gaps":[],"red_flags":[],"recommendation":"","candidate_feedback":[]}
overall_score — взвешенная сумма по весам JD (незаполненные навыки = 0).
Учитывайте блок "consistency": низкую достоверность как существенный red flag.
"""

# =============================
# State
# =============================
@dataclass
class SessionState:
    session_id: str
    jd_weights: Dict[str, float]
    resume_json: Dict[str, Any]
    resume_text: str
    rolling_summary: str = ""
    recent_turns: deque = field(default_factory=lambda: deque(maxlen=8))
    coverage_map: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    claims: List[Dict[str, Any]] = field(default_factory=list)
    claim_status: Dict[str, str] = field(default_factory=dict)  # id -> supported|refuted|unclear
    consistency_score: float = 1.0
    consistency_events: List[Dict[str, Any]] = field(default_factory=list)  # history of updates

# =============================
# Hooks (optional)
# =============================
def post_json(url: Optional[str], payload: dict, api_key_env: Optional[str] = None):
    if not url: return
    headers={"Content-Type":"application/json"}
    if api_key_env and os.getenv(api_key_env):
        headers["Authorization"]=f"Bearer {os.getenv(api_key_env)}"
    try: requests.post(url, headers=headers, json=payload, timeout=10)
    except Exception as e: print(f"[WARN] POST {url}: {e}")

def mirror_to_front_subtitles(text: str): post_json(os.getenv("FRONT_SUBS_URL"), {"text": text})
def mirror_to_tts(text: str): post_json(os.getenv("TTS_URL"), {"text": text})

# =============================
# Helpers
# =============================
def ensure_russian(text: str, model: Optional[str]=None) -> str:
    def _mostly_cyr(t):
        letters=[ch for ch in t if ch.isalpha()]
        if not letters: return True
        cy=sum('а'<=ch.lower()<='я' or ch.lower()=='ё' for ch in letters)
        return cy/max(1,len(letters))>=0.6
    if not text or _mostly_cyr(text): return text
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content":"Переведи на русский и верни только текст без комментариев:\n"+text}
    out=chat_completion([system,user],model=model,temperature=0.1,max_tokens=300)
    return out.strip()

def parse_weights(weights_str: str) -> Dict[str, float]:
    parts=[p.strip() for p in weights_str.split(",") if p.strip()]; d={}
    for p in parts:
        toks=p.split()
        if len(toks)>=2:
            skill=" ".join(toks[:-1])
            try: w=float(toks[-1])
            except Exception: continue
            d[skill]=w
    s=sum(d.values()) or 1.0
    return {k:v/s for k,v in d.items()}

def now_iso(): return time.strftime("%Y-%m-%dT%H:%M:%S")

# =============================
# Core steps
# =============================
def compress_resume(resume_text: str, model: Optional[str]=None) -> Dict[str, Any]:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content":RESUME_COMPRESS_PROMPT+"\n\nРЕЗЮМЕ:\n"+resume_text}
    out=chat_completion([system,user],model=model,temperature=0.1,max_tokens=800)
    return safe_json_loads(out)

def extract_claims(resume_text: str, resume_json: Dict[str, Any], model: Optional[str]=None) -> List[Dict[str, Any]]:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content":RESUME_CLAIMS_PROMPT+"\n\nИСХОДНОЕ_РЕЗЮМЕ:\n"+resume_text+"\n\nСЖАТОЕ_РЕЗЮМЕ:\n"+json.dumps(resume_json,ensure_ascii=False)}
    out=chat_completion([system,user],model=model,temperature=0.1,max_tokens=600)
    return safe_json_loads(out).get("claims", [])

def plan_questions(jd_weights: Dict[str, float], resume_json: Dict[str, Any],
                   model: Optional[str]=None, num_questions:int=5) -> Dict[str, Any]:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content":QUESTION_PLANNER_PROMPT+
          f"\n\nСделайте минимум {num_questions} вопросов (лучше ровно {num_questions})."+
          f"\n\nJD_WEIGHTS:\n{json.dumps(jd_weights,ensure_ascii=False)}"+
          f"\n\nRESUME_JSON:\n{json.dumps(resume_json,ensure_ascii=False)}"}
    out=chat_completion([system,user],model=model,temperature=0.1,max_tokens=700)
    return safe_json_loads(out)

def update_rolling_summary(prev_summary: str, role: str, text: str, model: Optional[str]=None) -> str:
    system={"role":"system","content":RU_SYSTEM}
    user={"role":"user","content": "Вы — RollingSummarizer. Обновите сводку на русском (<120 слов). Верните ТОЛЬКО JSON: {\"rolling_summary\":\"...\"}\n\n"
                                   + json.dumps({"prev_summary":prev_summary,"new_turn":{"role":role,"text":text}},ensure_ascii=False)}
    out=chat_completion([system,user],model=model,temperature=0.1,max_tokens=220)
    try: return safe_json_loads(out).get("rolling_summary",prev_summary)
    except Exception: return prev_summary

def turn_policy(state: SessionState, question_obj: Dict[str,Any], answer_text: str,
                model: Optional[str]=None) -> Dict[str,Any]:
    system={"role":"system","content":RU_SYSTEM}
    payload = {
        "jd_weights": state.jd_weights,
        "resume_json": state.resume_json,
        "resume_text": state.resume_text[:4000],  # safety crop
        "rolling_summary": state.rolling_summary,
        "recent_turns": list(state.recent_turns),
        "current_skill_scores": {k: state.coverage_map.get(k,{}).get("score",0.0) for k in state.jd_weights.keys()},
        "claims": state.claims,
        "claim_status": state.claim_status,
        "question_obj": question_obj,
        "answer_text": answer_text
    }
    user={"role":"user","content": TURN_POLICY_PROMPT + "\n\n" + json.dumps(payload, ensure_ascii=False)}
    out=chat_completion([system,user],model=model,temperature=0.1,max_tokens=900)
    return safe_json_loads(out)

def build_final_report(jd_weights: Dict[str, float], resume_json: Dict[str, Any],
                       evidence: Dict[str, List[str]], final_skill_scores: Dict[str, float],
                       consistency: Dict[str, Any],
                       model: Optional[str]=None) -> Dict[str, Any]:
    def _ensure_list(x):
        if isinstance(x,list): return x
        if x is None: return []
        if isinstance(x,str): return [x]
        try: return list(x)
        except Exception: return [x]
    enriched=dict(resume_json)
    ebs=dict(enriched.get("evidence_by_skill",{}))
    for sk,val in list(ebs.items()): ebs[sk]=_ensure_list(val)
    for k,v in (evidence or {}).items():
        v_list=_ensure_list(v); ebs.setdefault(k,[]); ebs[k]=_ensure_list(ebs[k])
        for it in v_list:
            if it not in ebs[k]: ebs[k].append(it)
    enriched["evidence_by_skill"]=ebs
    system={"role":"system","content":RU_SYSTEM}
    reporter_payload = {
        "jd_weights": jd_weights,
        "resume_json": enriched,
        "final_skill_scores": final_skill_scores,
        "consistency": consistency
    }
    user={"role":"user","content":FINAL_REPORT_PROMPT+"\n\n"+json.dumps(reporter_payload,ensure_ascii=False)}
    out=chat_completion([system,user],model=model,temperature=0.1,max_tokens=900)
    return safe_json_loads(out)

# =============================
# Interactive loop with disqualify
# =============================
@dataclass
class LoopResult:
    evidence: Dict[str, List[str]]
    skill_scores: Dict[str, float]
    report: Dict[str, Any]
    rolling_summary: str
    coverage_map: Dict[str, Dict[str, Any]]

def loop(state: SessionState, model: Optional[str]=None,
         max_turns:int=6, min_turns:int=2,
         consistency_threshold: float=0.45, max_inconsistency: int=2, min_probe:int=2) -> LoopResult:
    plan=plan_questions(state.jd_weights,state.resume_json,model=model,num_questions=max_turns)
    queue=[q for q in plan.get("prioritized_questions",[]) if q.get("question")]
    evidence: Dict[str, List[str]] = {}
    scores: Dict[str, float] = {k: 0.0 for k in state.jd_weights.keys()}

    i=0; first=True; probed=False
    while i<max_turns:
        if not queue:
            extra=plan_questions(state.jd_weights,state.resume_json,model=model,num_questions=(max_turns-i))
            queue=[q for q in extra.get("prioritized_questions",[]) if q.get("question")]
            if not queue: break

        q=queue.pop(0); skill=q.get("skill","General")
        q_text = ensure_russian(q.get("question",""), model=model)

        if first:
            print("👋 Добрый день! Спасибо, что нашли время на интервью. Задам несколько вопросов, чтобы лучше понять ваш опыт.")
            first=False
        else:
            print("\nСпасибо за ответ!")

        print(f"\n[Вопрос] ({skill}): {q_text}")
        if q.get("reason"):
            print(f"[Почему]: {ensure_russian(q.get('reason'), model=model)}")

        mirror_to_front_subtitles(q_text); mirror_to_tts(q_text)
        post_json(os.getenv("DRF_TURN_URL"), {
            "session_id": os.getenv("SESSION_ID","local"),
            "turn_id": str(uuid.uuid4()), "role": "agent", "text": q_text,
            "ts": now_iso(), "nlu_extract": {}, "coverage_delta": {}
        }, api_key_env="DRF_API_KEY")

        state.recent_turns.append({"role":"agent","text":q_text})
        state.rolling_summary=update_rolling_summary(state.rolling_summary,"agent",q_text,model)

        ans=input("\n[Ответ кандидата]: ").strip()
        state.recent_turns.append({"role":"candidate","text":ans})
        state.rolling_summary=update_rolling_summary(state.rolling_summary,"candidate",ans,model)

        pol = turn_policy(state, q, ans, model=model)

        pref = ensure_russian(pol.get("agent_preface_ru","") or "", model=model)
        if pref:
            print("\n" + pref)
            mirror_to_front_subtitles(pref); mirror_to_tts(pref)
            post_json(os.getenv("DRF_TURN_URL"), {"session_id": os.getenv("SESSION_ID","local"),
                "turn_id": str(uuid.uuid4()), "role": "agent", "text": pref, "ts": now_iso(),
                "nlu_extract": {}, "coverage_delta": {}}, api_key_env="DRF_API_KEY")

        comment = ensure_russian(pol.get("comment_for_candidate_ru","") or "", model=model)
        if comment:
            print(comment)
            mirror_to_front_subtitles(comment); mirror_to_tts(comment)
            post_json(os.getenv("DRF_TURN_URL"), {"session_id": os.getenv("SESSION_ID","local"),
                "turn_id": str(uuid.uuid4()), "role": "agent", "text": comment, "ts": now_iso(),
                "nlu_extract": {}, "coverage_delta": {}}, api_key_env="DRF_API_KEY")

        # scores & evidence
        for k,v in (pol.get("skill_scores") or {}).items():
            try: scores[k]=float(v)
            except Exception: pass
        new_evs = pol.get("new_evidence")
        if isinstance(new_evs,str): new_evs=[new_evs]
        for ev in (new_evs or []): evidence.setdefault(skill,[]).append(ev)

        prev = state.coverage_map.get(skill, {"asked":0,"answered":0,"score":0.0})
        sc = float(pol.get("addressed_score", scores.get(skill,0.0)) or 0.0)
        state.coverage_map[skill] = {"asked":prev["asked"]+1,"answered":prev["answered"]+1,"score":sc}
        scores[skill] = sc

        # claim updates & consistency
        for upd in (pol.get("claim_updates") or []):
            cid = upd.get("id")
            st = upd.get("status")
            if cid and st:
                state.claim_status[cid] = st
                state.consistency_events.append({"id": cid, "status": st, "evidence": upd.get("evidence","")})

        # EMA-like update of consistency score (prior 1.0, weight 0.5 current)
        cur_cons = float(pol.get("consistency_score", state.consistency_score) or 0.0)
        state.consistency_score = max(0.0, min(1.0, 0.5*state.consistency_score + 0.5*cur_cons))

        # count refuted high-critical claims
        refuted_critical = 0
        crit_by_id = {c.get("id"): c.get("criticality","M") for c in state.claims}
        for cid, st in state.claim_status.items():
            if st == "refuted" and (crit_by_id.get(cid) in ("H","M")):
                refuted_critical += 1

        # candidate turn -> DRF
        post_json(os.getenv("DRF_TURN_URL"), {
            "session_id": os.getenv("SESSION_ID","local"),
            "turn_id": str(uuid.uuid4()), "role": "candidate", "text": ans,
            "ts": now_iso(),
            "nlu_extract": {
                "new_evidence": new_evs or [],
                "red_flags": pol.get("red_flags", []),
                "contradictions": pol.get("contradictions", []),
                "claim_updates": pol.get("claim_updates", []),
                "consistency_score": state.consistency_score,
                "skill_scores": scores
            },
            "coverage_delta": {skill: state.coverage_map[skill]}
        }, api_key_env="DRF_API_KEY")

        # next action + dynamic disqualify
        na = pol.get("next_action","next_topic")
        disq = bool(pol.get("disqualify", False))
        reason = ensure_russian(pol.get("disqualification_reason_ru","") or "", model=model)

        # условия раннего завершения
        early_end = False
        if i >= (min_probe-1) and (disq or state.consistency_score < consistency_threshold or refuted_critical >= max_inconsistency):
            # попробуем мягкую проверку один раз, если модель дала
            probe = pol.get("consistency_probe_ru")
            if probe and not probed:
                fu = ensure_russian(probe, model=model)
                print(f"\nПозвольте уточнить: {fu}")
                mirror_to_front_subtitles(fu); mirror_to_tts(fu)
                post_json(os.getenv("DRF_TURN_URL"), {"session_id": os.getenv("SESSION_ID","local"),
                    "turn_id": str(uuid.uuid4()), "role": "agent", "text": fu, "ts": now_iso(),
                    "nlu_extract": {}, "coverage_delta": {}}, api_key_env="DRF_API_KEY")
                state.recent_turns.append({"role":"agent","text":fu})
                state.rolling_summary=update_rolling_summary(state.rolling_summary,"agent",fu,model)

                ans2=input("\n[Ответ кандидата]: ").strip()
                state.recent_turns.append({"role":"candidate","text":ans2})
                state.rolling_summary=update_rolling_summary(state.rolling_summary,"candidate",ans2,model)

                pol2 = turn_policy(state, q, ans2, model=model)
                # обновить метрики по проверке
                for upd in (pol2.get("claim_updates") or []):
                    cid = upd.get("id"); st = upd.get("status")
                    if cid and st:
                        state.claim_status[cid] = st
                        state.consistency_events.append({"id": cid, "status": st, "evidence": upd.get("evidence","")})
                cur_cons2 = float(pol2.get("consistency_score", state.consistency_score) or 0.0)
                state.consistency_score = max(0.0, min(1.0, 0.5*state.consistency_score + 0.5*cur_cons2))

                post_json(os.getenv("DRF_TURN_URL"), {"session_id": os.getenv("SESSION_ID","local"),
                    "turn_id": str(uuid.uuid4()), "role": "candidate", "text": ans2, "ts": now_iso(),
                    "nlu_extract": {"consistency_score": state.consistency_score, "claim_updates": pol2.get("claim_updates", [])},
                    "coverage_delta": {}}, api_key_env="DRF_API_KEY")

                probed = True
                # переоценим условия
                refuted_critical = 0
                for cid, st in state.claim_status.items():
                    if st == "refuted" and (crit_by_id.get(cid) in ("H","M")):
                        refuted_critical += 1
                disq = bool(pol2.get("disqualify", disq))
                reason = ensure_russian(pol2.get("disqualification_reason_ru","") or reason, model=model)

            # финальная проверка порогов
            if disq or state.consistency_score < consistency_threshold or refuted_critical >= max_inconsistency:
                early_end = True

        if early_end:
            if reason:
                print("\n[Система]: Благодарю вас за время. Похоже, что заявленные в резюме сведения не подтверждаются. " +
                      f"Поэтому мы завершим интервью. Короткое пояснение: {reason}")
            else:
                print("\n[Система]: Благодарю вас за время. Похоже, что заявленные в резюме сведения не подтверждаются, поэтому мы завершим интервью.")
            break

        if na == "end" and i < (max(min_turns,1)-1):
            na = "next_topic"

        if na == "ask_followup" and pol.get("followup_question_ru"):
            fu = ensure_russian(pol["followup_question_ru"], model=model)
            print(f"\nУточню: {fu}")
            mirror_to_front_subtitles(fu); mirror_to_tts(fu)
            post_json(os.getenv("DRF_TURN_URL"), {"session_id": os.getenv("SESSION_ID","local"),
                "turn_id": str(uuid.uuid4()), "role": "agent", "text": fu, "ts": now_iso(),
                "nlu_extract": {}, "coverage_delta": {}}, api_key_env="DRF_API_KEY")
            state.recent_turns.append({"role":"agent","text":fu})
            state.rolling_summary=update_rolling_summary(state.rolling_summary,"agent",fu,model)

            ans2=input("\n[Ответ кандидата]: ").strip()
            state.recent_turns.append({"role":"candidate","text":ans2})
            state.rolling_summary=update_rolling_summary(state.rolling_summary,"candidate",ans2,model)

            pol2 = turn_policy(state, q, ans2, model=model)

            for k,v in (pol2.get("skill_scores") or {}).items():
                try: scores[k]=float(v)
                except Exception: pass
            new_evs2 = pol2.get("new_evidence")
            if isinstance(new_evs2,str): new_evs2=[new_evs2]
            for ev in (new_evs2 or []): evidence.setdefault(skill,[]).append(ev)

            state.coverage_map[skill]["answered"] += 1
            state.coverage_map[skill]["score"] = float(pol2.get("addressed_score", scores.get(skill,0.0)) or 0.0)

            post_json(os.getenv("DRF_TURN_URL"), {"session_id": os.getenv("SESSION_ID","local"),
                "turn_id": str(uuid.uuid4()), "role": "candidate", "text": ans2, "ts": now_iso(),
                "nlu_extract": {"new_evidence": new_evs2 or [], "skill_scores": scores},
                "coverage_delta": {skill: state.coverage_map[skill]}}, api_key_env="DRF_API_KEY")

            na = pol2.get("next_action","next_topic")
            if na == "end" and i < (max(min_turns,1)-1):
                na = "next_topic"

        if na == "next_topic" and pol.get("next_topic_question_ru"):
            queue.insert(0, {"skill":"(policy)","question": pol["next_topic_question_ru"], "reason":"policy-suggested"})

        if na == "end":
            print("\n[Система]: Спасибо, что уделили время. На этом всё.")
            break

        i+=1

    # Build consistency summary for report
    supported = [e for e in state.consistency_events if e.get("status")=="supported"]
    refuted   = [e for e in state.consistency_events if e.get("status")=="refuted"]
    consistency = {
        "score": state.consistency_score,
        "supported": supported[:20],
        "refuted": refuted[:20]
    }

    report=build_final_report(state.jd_weights,state.resume_json,evidence,scores,consistency,model)
    return LoopResult(evidence=evidence,skill_scores=scores,report=report,
                      rolling_summary=state.rolling_summary,coverage_map=state.coverage_map)

# =============================
# CLI
# =============================
def main():
    ap=argparse.ArgumentParser(description="HR Avatar — RU dynamic with claim-check & early end")
    ap.add_argument("--session-id",type=str,default=str(uuid.uuid4()))
    ap.add_argument("--resume",type=str,default="sample_resume.txt")
    ap.add_argument("--jd",type=str,default="sample_jd.txt")
    ap.add_argument("--weights",type=str,default="")
    ap.add_argument("--model",type=str,default=None)
    ap.add_argument("--max-turns",type=int,default=3)
    ap.add_argument("--min-turns",type=int,default=2)
    ap.add_argument("--consistency-threshold",type=float,default=0.45)
    ap.add_argument("--max-inconsistency",type=int,default=2)
    ap.add_argument("--min-probe",type=int,default=2)
    ap.add_argument("--compress-only",action="store_true")
    a=ap.parse_args()

    os.environ["SESSION_ID"]=a.session_id

    with open(a.resume,"r",encoding="utf-8") as f:
        resume_text = f.read()
    with open(a.jd,"r",encoding="utf-8") as f:
        jd_text = f.read()

    if a.weights.strip():
        jw=parse_weights(a.weights)
    else:
        system={"role":"system","content":RU_SYSTEM}
        user={"role":"user","content":"На русском: выделите 4–7 ключевых компетенций из JD с весами (сумма 1.0). "
                                     "Верните ТОЛЬКО JSON вида {\"weights\": {\"Навык\": вес}}.\n\nJD:\n"+jd_text}
        out=chat_completion([system,user],model=a.model,temperature=0.1,max_tokens=500)
        jw=safe_json_loads(out).get("weights",{}); s=sum(jw.values()) or 1.0; jw={k:v/s for k,v in jw.items()}

    rj=compress_resume(resume_text,model=a.model)
    claims=extract_claims(resume_text,rj,model=a.model)

    if a.compress_only:
        print(json.dumps({"resume_json":rj,"claims":claims},ensure_ascii=False,indent=2)); return

    state=SessionState(session_id=a.session_id,jd_weights=jw,resume_json=rj,resume_text=resume_text,claims=claims)

    print("\n=== Сессия:",a.session_id,"===")
    print("=== Приоритеты JD ===")
    for k,v in jw.items(): print(f"- {k}: {v:.2f}")
    print("\n=== Резюме (сжатое) ===")
    print(json.dumps(rj,ensure_ascii=False,indent=2))
    print("\n=== Проверяемые утверждения (claims) ===")
    for c in claims[:10]: print(f"- {c.get('id')}: {c.get('text')} (критичность {c.get('criticality')})")

    res=loop(state,model=a.model,max_turns=a.max_turns,min_turns=a.min_turns,
             consistency_threshold=a.consistency_threshold,max_inconsistency=a.max_inconsistency,min_probe=a.min_probe)

    ts=int(time.time()); out=f"runs/run_{ts}"; os.makedirs(out,exist_ok=True)
    open(out+"/jd_weights.json","w",encoding="utf-8").write(json.dumps(jw,ensure_ascii=False,indent=2))
    open(out+"/resume_compressed.json","w",encoding="utf-8").write(json.dumps(rj,ensure_ascii=False,indent=2))
    open(out+"/claims.json","w",encoding="utf-8").write(json.dumps(claims,ensure_ascii=False,indent=2))
    open(out+"/skill_scores.json","w",encoding="utf-8").write(json.dumps(res.skill_scores,ensure_ascii=False,indent=2))
    open(out+"/evidence.json","w",encoding="utf-8").write(json.dumps(res.evidence,ensure_ascii=False,indent=2))
    open(out+"/coverage_map.json","w",encoding="utf-8").write(json.dumps(res.coverage_map,ensure_ascii=False,indent=2))
    open(out+"/report.json","w",encoding="utf-8").write(json.dumps(res.report,ensure_ascii=False,indent=2))

    lines=["# Итоговый отчёт об интервью",
           f"**Общий скор**: {res.report.get('overall_score',0):.2f} — **Решение**: {res.report.get('decision','')}",
           "","## Разбивка по навыкам"]
    for sbr in res.report.get("skills_breakdown",[]): lines.append(f"- **{sbr.get('skill')}**: {sbr.get('score',0):.2f} (вес {sbr.get('weight')}) — доказательства: "+"; ".join(sbr.get('evidence') or []))
    if res.report.get("strengths"): lines.append("\n## Сильные стороны"); [lines.append("- "+x) for x in res.report["strengths"]]
    if res.report.get("gaps"): lines.append("\n## Пробелы"); [lines.append("- "+x) for x in res.report["gaps"]]
    if res.report.get("red_flags"): lines.append("\n## Red flags"); [lines.append("- "+x) for x in res.report["red_flags"]]
    if res.report.get("recommendation"): lines.append("\n## Рекомендация"); lines.append(res.report["recommendation"])
    if res.report.get("candidate_feedback"):
        lines.append("\n## Обратная связь кандидату"); [lines.append("- "+x) for x in res.report["candidate_feedback"]]
    open(out+"/report.md","w",encoding="utf-8").write("\n".join(lines))

    print("\n[Система]: Благодарю за беседу! Отчёт подготовлен.")
    print(f"Готово! Артефакты сохранены в: {out}")
    print("Основное: report.json и report.md")
    print("Флаги управления: --consistency-threshold, --max-inconsistency, --min-probe")

if __name__=="__main__": main()
