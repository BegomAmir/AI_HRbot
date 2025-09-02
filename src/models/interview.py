"""
Модели данных для интервью и оценки кандидатов
"""
from typing import List, Dict, Optional, Any
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field


class InterviewStatus(str, Enum):
    """Статусы интервью"""
    SCHEDULED = "scheduled"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class EvaluationResult(str, Enum):
    """Результаты оценки"""
    PASS = "pass"
    FAIL = "fail"
    NEEDS_CLARIFICATION = "needs_clarification"


class SkillLevel(str, Enum):
    """Уровни навыков"""
    BEGINNER = "beginner"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class AudioSegment(BaseModel):
    """Сегмент аудио с метаданными"""
    start_time: float
    end_time: float
    duration: float
    audio_data: bytes
    confidence: float = 0.0
    is_speech: bool = True


class SpeechTranscription(BaseModel):
    """Транскрипция речи"""
    text: str
    confidence: float
    language: str
    start_time: float
    end_time: float
    audio_segment: AudioSegment


class ProsodyFeatures(BaseModel):
    """Просодические характеристики речи"""
    pitch_mean: float
    pitch_std: float
    energy_mean: float
    energy_std: float
    speaking_rate: float
    pause_duration: float
    voice_quality: str


class EmotionAnalysis(BaseModel):
    """Анализ эмоций в речи"""
    primary_emotion: str
    confidence: float
    emotions_distribution: Dict[str, float]
    valence: float  # -1 (негатив) до 1 (позитив)
    arousal: float  # 0 (спокойствие) до 1 (возбуждение)


class SkillAssessment(BaseModel):
    """Оценка навыка"""
    skill_name: str
    required_level: SkillLevel
    candidate_level: SkillLevel
    confidence: float
    evidence: List[str]
    score: float  # 0-100


class CommunicationAssessment(BaseModel):
    """Оценка коммуникативных навыков"""
    clarity: float  # 0-100
    fluency: float  # 0-100
    engagement: float  # 0-100
    listening_skills: float  # 0-100
    overall_score: float  # 0-100
    notes: List[str]


class CandidateResponse(BaseModel):
    """Ответ кандидата на вопрос"""
    question_id: str
    question_text: str
    response_text: str
    transcription: SpeechTranscription
    prosody: ProsodyFeatures
    emotions: EmotionAnalysis
    processing_time: float
    timestamp: datetime


class InterviewQuestion(BaseModel):
    """Вопрос интервью"""
    id: str
    text: str
    category: str
    weight: float
    follow_up_questions: List[str] = []
    required_skills: List[str] = []
    difficulty: str = "medium"


class InterviewSession(BaseModel):
    """Сессия интервью"""
    id: str
    candidate_id: str
    vacancy_id: str
    status: InterviewStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    questions: List[InterviewQuestion]
    responses: List[CandidateResponse] = []
    current_question_index: int = 0
    total_score: float = 0.0
    evaluation_result: Optional[EvaluationResult] = None


class CandidateEvaluation(BaseModel):
    """Полная оценка кандидата"""
    candidate_id: str
    vacancy_id: str
    interview_session_id: str
    
    # Оценки по категориям
    technical_skills: List[SkillAssessment]
    communication: CommunicationAssessment
    experience: Dict[str, Any]
    
    # Общие метрики
    overall_score: float
    confidence: float
    red_flags: List[str]
    strong_points: List[str]
    improvement_areas: List[str]
    
    # Рекомендации
    recommendation: EvaluationResult
    next_steps: List[str]
    feedback_for_candidate: str
    
    # Метаданные
    evaluation_timestamp: datetime
    processing_duration: float
    model_version: str


class InterviewMetrics(BaseModel):
    """Метрики интервью для отправки в Django"""
    session_id: str
    candidate_id: str
    vacancy_id: str
    transcriptions: List[SpeechTranscription]
    prosody_features: List[ProsodyFeatures]
    emotion_analysis: List[EmotionAnalysis]
    skill_assessments: List[SkillAssessment]
    communication_score: float
    overall_evaluation: CandidateEvaluation
    timestamp: datetime
