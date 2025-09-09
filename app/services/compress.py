from typing import Dict, Any, List, Optional
from ..clients.llm import LLMClient
from ..core.prompts import RU_SYSTEM, RESUME_COMPRESS_PROMPT, RESUME_CLAIMS_PROMPT

class ResumeService:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    async def compress(self, resume_text: str, model: Optional[str]=None) -> Dict[str,Any]:
        sys={"role":"system","content":RU_SYSTEM}
        usr={"role":"user","content":RESUME_COMPRESS_PROMPT+"\n\nРЕЗЮМЕ:\n"+resume_text}
        out = await self.llm.chat_completion([sys,usr], temperature=0.1, max_tokens=800)
        return self.llm.safe_json_loads(out)

    async def extract_claims(self, resume_text: str, resume_json: Dict[str, Any], model: Optional[str]=None) -> List[Dict[str,Any]]:
        sys={"role":"system","content":RU_SYSTEM}
        usr={"role":"user","content":RESUME_CLAIMS_PROMPT+"\n\nИСХОДНОЕ_РЕЗЮМЕ:\n"+resume_text+"\n\nСЖАТОЕ_РЕЗЮМЕ:\n"+str(resume_json)}
        out = await self.llm.chat_completion([sys,usr], temperature=0.1, max_tokens=600)
        return self.llm.safe_json_loads(out).get("claims", [])
