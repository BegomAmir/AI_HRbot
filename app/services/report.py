from typing import Dict, Any
from ..clients.llm import LLMClient
from ..core.prompts import RU_SYSTEM, FINAL_REPORT_PROMPT

class ReportService:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def build_report(self, payload: Dict[str,Any]) -> Dict[str,Any]:
        sys={"role":"system","content":RU_SYSTEM}
        usr={"role":"user","content": FINAL_REPORT_PROMPT + "\n\n" + str(payload)}
        out = await self.llm.chat_completion([sys,usr], temperature=0.1, max_tokens=900)
        return self.llm.safe_json_loads(out)
