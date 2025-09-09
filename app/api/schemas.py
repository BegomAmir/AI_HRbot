from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

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
