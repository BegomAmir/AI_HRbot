from typing import Dict, Any, List, Optional
from ..clients.llm import LLMClient
from ..core.prompts import RU_SYSTEM, QUESTION_PLANNER_PROMPT, TURN_POLICY_PROMPT

class InterviewService:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def plan_questions(self, jd_weights: Dict[str,float], resume_json: Dict[str,Any], num_questions:int=5) -> Dict[str,Any]:
        sys={"role":"system","content":RU_SYSTEM}
        usr={"role":"user","content": QUESTION_PLANNER_PROMPT +
             f"\n\nСделайте минимум {num_questions} вопросов (лучше ровно {num_questions})." +
             f"\n\nJD_WEIGHTS:\n{jd_weights}" +
             f"\n\nRESUME_JSON:\n{resume_json}" }
        out = await self.llm.chat_completion([sys,usr], temperature=0.1, max_tokens=700)
        return self.llm.safe_json_loads(out)

    async def rolling_summary(self, prev_summary: str, role: str, text: str) -> str:
        sys={"role":"system","content":RU_SYSTEM}
        usr={"role":"user","content":
             "Вы — RollingSummarizer. Обновите сводку на русском (<120 слов). Верните ТОЛЬКО JSON: {\"rolling_summary\":\"...\"}\n\n"
             + str({"prev_summary":prev_summary,"new_turn":{"role":role,"text":text}})
        }
        out = await self.llm.chat_completion([sys,usr], temperature=0.1, max_tokens=220)
        try:
            return self.llm.safe_json_loads(out).get("rolling_summary", prev_summary)
        except Exception:
            return prev_summary

    async def turn_policy(self, payload: Dict[str,Any]) -> Dict[str,Any]:
        sys={"role":"system","content":RU_SYSTEM}
        usr={"role":"user","content": TURN_POLICY_PROMPT + "\n\n" + str(payload)}
        out = await self.llm.chat_completion([sys,usr], temperature=0.1, max_tokens=900)
        return self.llm.safe_json_loads(out)
