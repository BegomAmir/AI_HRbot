from fastapi import APIRouter, Request
from ..api.schemas import *
from ..services.compress import ResumeService
from ..services.interview import InterviewService
from ..services.report import ReportService

router = APIRouter()

class Container:
    def __init__(self, resume: ResumeService, interview: InterviewService, report: ReportService):
        self.resume = resume
        self.interview = interview
        self.report = report

@router.post("/compress_resume", response_model=CompressResp)
async def compress_resume_ep(req: CompressReq, request: Request):
    c: Container = request.app.state.container
    rj = await c.resume.compress(req.resume, req.model)
    claims = await c.resume.extract_claims(req.resume, rj, req.model) if req.with_claims else None
    return CompressResp(resume_compressed=rj, claims=claims)

@router.post("/interact_stateless", response_model=InteractResp)
async def interact_stateless_ep(req: InteractReq, request: Request):
    c: Container = request.app.state.container
    st = req.state
    if not st.last_question:
        plan = await c.interview.plan_questions(req.jd_weights, req.resume_compressed, num_questions=req.num_questions)
        st.question_queue = [q for q in plan.get("prioritized_questions",[]) if q.get("question")]
        st.last_question = st.question_queue.pop(0) if st.question_queue else None
        return InteractResp(next_question=st.last_question, finished=False, reason=None, state=st)
    st.recent_turns.append({"role":"agent","text": st.last_question.get("question","")})
    st.rolling_summary = await c.interview.rolling_summary(st.rolling_summary, "agent", st.last_question.get("question",""))
    st.recent_turns.append({"role":"candidate","text": req.candidate_text or ""})
    st.rolling_summary = await c.interview.rolling_summary(st.rolling_summary, "candidate", req.candidate_text or "")
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
    pol = await c.interview.turn_policy(payload)
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
                    plan = await c.interview.plan_questions(req.jd_weights, req.resume_compressed, num_questions=req.num_questions)
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

@router.post("/final_stateless", response_model=FinalResp)
async def final_stateless_ep(req: FinalReq, request: Request):
    c: Container = request.app.state.container
    final_scores = {k: req.coverage_map.get(k,{}).get("score",0.0) for k in req.jd_weights.keys()}
    consistency = {"score": req.consistency_score,
                   "supported": req.consistency_supported[:20],
                   "refuted": req.consistency_refuted[:20]}
    payload = {"jd_weights": req.jd_weights, "resume_json": req.resume_compressed,
               "final_skill_scores": final_scores, "consistency": consistency, "evidence": req.evidence}
    report = await c.report.build_report(payload)
    return FinalResp(report=report)
