import os, json, re, httpx
from typing import List, Dict, Any, Optional

class LLMClient:
    """Async OpenAI-compatible chat client"""
    def __init__(self, client: httpx.AsyncClient, model: Optional[str]=None, base_url: Optional[str]=None, api_key: Optional[str]=None):
        self.client = client
        self.model = model or os.getenv("LLM_MODEL", "openai/gpt-oss-20b")
        self.base_url = (base_url or os.getenv("LLM_BASE_URL","http://127.0.0.1:1234")).rstrip("/")
        self.api_key = api_key or os.getenv("LLM_API_KEY", "lm-studio")

    async def chat_completion(self, messages: List[Dict[str,Any]], temperature: float = 0.2, max_tokens: int = 700) -> str:
        url = f"{self.base_url}/v1/chat/completions"
        headers = {"Content-Type":"application/json; charset=utf-8"}
        if self.api_key: headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
        r = await self.client.post(url, headers=headers, json=payload, timeout=120)
        r.raise_for_status()
        j = r.json()
        return j["choices"][0]["message"]["content"]

    @staticmethod
    def safe_json_loads(s: str) -> Dict[str,Any]:
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
            try: return json.loads(c)
            except Exception: pass
            try: return json.loads(c.replace("'", '"'))
            except Exception: pass
        raise ValueError("LLM returned non-JSON")
