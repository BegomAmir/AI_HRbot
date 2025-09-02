"""
Конфигурация для Python Agent AI HR Bot
"""
import os
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Основные настройки приложения"""
    
    # LiveKit настройки
    LIVEKIT_URL: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    LIVEKIT_API_KEY: str = os.getenv("LIVEKIT_API_KEY", "")
    LIVEKIT_API_SECRET: str = os.getenv("LIVEKIT_API_SECRET", "")
    
    # Django API настройки
    DJANGO_API_URL: str = os.getenv("DJANGO_API_URL", "http://localhost:8000")
    DJANGO_API_TOKEN: str = os.getenv("DJANGO_API_TOKEN", "")
    
    # Аудио настройки
    AUDIO_SAMPLE_RATE: int = 16000
    AUDIO_CHANNELS: int = 1
    AUDIO_CHUNK_SIZE: int = 1024
    AUDIO_FORMAT: str = "wav"
    
    # STT настройки
    STT_MODEL_SIZE: str = "base"  # tiny, base, small, medium, large
    STT_LANGUAGE: str = "ru"  # ru, en, auto
    
    # VAD настройки
    VAD_THRESHOLD: float = 0.5
    VAD_MIN_SPEECH_DURATION: float = 0.5
    VAD_MAX_SPEECH_DURATION: float = 30.0
    
    # NLP настройки
    NLP_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    SPACY_MODEL: str = "ru_core_news_sm"
    
    # TTS настройки
    TTS_VOICE: str = "ru_ru-oksana-medium"
    TTS_SPEED: float = 1.0
    
    # Веса для оценки кандидатов
    EVALUATION_WEIGHTS: dict = {
        "technical_skills": 0.5,
        "communication": 0.3,
        "experience": 0.2
    }
    
    # Логирование
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    class Config:
        env_file = ".env"


# Создаем глобальный экземпляр настроек
settings = Settings()
