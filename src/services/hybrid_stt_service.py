"""
Гибридный STT сервис
Автоматически выбирает между WhisperX и faster-whisper в зависимости от доступности
"""
import asyncio
from typing import List, Optional, Dict, Any
from src.models.interview import AudioSegment, SpeechTranscription
from src.services.stt_service import STTService
from src.services.whisperx_stt_service import WhisperXSTTService
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class HybridSTTService:
    """
    Гибридный STT сервис
    
    Автоматически выбирает лучший доступный движок:
    1. WhisperX (если доступен) - для высокой точности и диаризации
    2. faster-whisper (fallback) - для быстрой работы
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.HybridSTTService")
        
        # Инициализируем оба сервиса
        self.whisperx_service = WhisperXSTTService()
        self.faster_whisper_service = STTService()
        
        # Определяем основной сервис
        self.primary_service = None
        self.fallback_service = None
        self._select_primary_service()
    
    def _select_primary_service(self):
        """Выбор основного STT сервиса"""
        try:
            # Проверяем доступность WhisperX
            if self.whisperx_service.model is not None:
                self.primary_service = self.whisperx_service
                self.fallback_service = self.faster_whisper_service
                self.logger.info("WhisperX выбран как основной STT сервис")
            else:
                self.primary_service = self.faster_whisper_service
                self.fallback_service = None
                self.logger.info("faster-whisper выбран как основной STT сервис")
                
        except Exception as e:
            self.logger.error(f"Ошибка выбора STT сервиса: {e}")
            self.primary_service = self.faster_whisper_service
            self.fallback_service = None
    
    async def transcribe_audio(
        self, 
        audio_segment: AudioSegment,
        language: Optional[str] = None,
        use_advanced_features: bool = True
    ) -> Optional[SpeechTranscription]:
        """
        Транскрипция аудио с автоматическим выбором движка
        
        Args:
            audio_segment: Аудио сегмент для транскрипции
            language: Язык речи
            use_advanced_features: Использовать продвинутые функции (диаризация, выравнивание)
            
        Returns:
            Транскрипция речи или None при ошибке
        """
        try:
            if self.primary_service is None:
                self.logger.error("Нет доступных STT сервисов")
                return None
            
            # Пытаемся использовать основной сервис
            try:
                if isinstance(self.primary_service, WhisperXSTTService) and use_advanced_features:
                    # Используем WhisperX с продвинутыми функциями
                    result = await self.primary_service.transcribe_audio(
                        audio_segment, 
                        language,
                        enable_diarization=True,
                        enable_word_alignment=True
                    )
                else:
                    # Используем стандартную транскрипцию
                    result = await self.primary_service.transcribe_audio(audio_segment, language)
                
                if result:
                    self.logger.debug(f"Транскрипция выполнена с {type(self.primary_service).__name__}")
                    return result
                    
            except Exception as e:
                self.logger.warning(f"Ошибка основного STT сервиса: {e}")
                
                # Пробуем fallback сервис
                if self.fallback_service is not None:
                    try:
                        result = await self.fallback_service.transcribe_audio(audio_segment, language)
                        if result:
                            self.logger.info("Использован fallback STT сервис")
                            return result
                    except Exception as fallback_error:
                        self.logger.error(f"Ошибка fallback STT сервиса: {fallback_error}")
            
            return None
            
        except Exception as e:
            self.logger.error(f"Критическая ошибка транскрипции: {e}")
            return None
    
    async def transcribe_with_speakers(
        self, 
        audio_segment: AudioSegment,
        language: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Транскрипция с разделением спикеров
        
        Args:
            audio_segment: Аудио сегмент
            language: Язык речи
            
        Returns:
            Результат с разделением спикеров или None
        """
        try:
            if isinstance(self.primary_service, WhisperXSTTService):
                return await self.primary_service.transcribe_with_speakers(audio_segment, language)
            else:
                self.logger.warning("Диаризация доступна только с WhisperX")
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка транскрипции с спикерами: {e}")
            return None
    
    async def transcribe_batch(
        self, 
        audio_segments: List[AudioSegment],
        language: Optional[str] = None
    ) -> List[SpeechTranscription]:
        """
        Пакетная транскрипция
        
        Args:
            audio_segments: Список аудио сегментов
            language: Язык речи
            
        Returns:
            Список транскрипций
        """
        transcriptions = []
        
        for segment in audio_segments:
            transcription = await self.transcribe_audio(segment, language)
            if transcription:
                transcriptions.append(transcription)
        
        return transcriptions
    
    async def transcribe_stream(
        self, 
        audio_stream: List[AudioSegment],
        language: Optional[str] = None,
        callback: Optional[callable] = None
    ) -> List[SpeechTranscription]:
        """
        Потоковая транскрипция
        
        Args:
            audio_stream: Поток аудио сегментов
            language: Язык речи
            callback: Колбэк для обработки каждой транскрипции
            
        Returns:
            Список всех транскрипций
        """
        transcriptions = []
        
        for segment in audio_stream:
            transcription = await self.transcribe_audio(segment, language)
            if transcription:
                transcriptions.append(transcription)
                
                if callback:
                    await callback(transcription)
        
        return transcriptions
    
    def get_supported_languages(self) -> List[str]:
        """Получение списка поддерживаемых языков"""
        if self.primary_service:
            return self.primary_service.get_supported_languages()
        return []
    
    def detect_language(self, audio_segment: AudioSegment) -> Optional[str]:
        """
        Автоопределение языка речи
        
        Args:
            audio_segment: Аудио сегмент
            
        Returns:
            Код языка или None
        """
        try:
            if hasattr(self.primary_service, 'detect_language'):
                return self.primary_service.detect_language(audio_segment)
            return None
        except Exception as e:
            self.logger.error(f"Ошибка определения языка: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о моделях"""
        info = {
            "primary_service": type(self.primary_service).__name__ if self.primary_service else None,
            "fallback_service": type(self.fallback_service).__name__ if self.fallback_service else None,
            "whisperx_available": isinstance(self.primary_service, WhisperXSTTService),
            "faster_whisper_available": isinstance(self.primary_service, STTService) or self.fallback_service is not None
        }
        
        # Добавляем детальную информацию от основного сервиса
        if self.primary_service and hasattr(self.primary_service, 'get_model_info'):
            primary_info = self.primary_service.get_model_info()
            info.update({"primary_details": primary_info})
        
        return info
    
    async def health_check(self) -> bool:
        """Проверка работоспособности сервиса"""
        try:
            if self.primary_service is None:
                return False
            
            # Проверяем основной сервис
            primary_healthy = await self.primary_service.health_check()
            
            # Проверяем fallback сервис
            fallback_healthy = False
            if self.fallback_service and hasattr(self.fallback_service, 'health_check'):
                fallback_healthy = await self.fallback_service.health_check()
            
            return primary_healthy or fallback_healthy
            
        except Exception as e:
            self.logger.error(f"Ошибка health check: {e}")
            return False
    
    def get_capabilities(self) -> Dict[str, bool]:
        """Получение возможностей сервиса"""
        return {
            "basic_transcription": True,
            "word_alignment": isinstance(self.primary_service, WhisperXSTTService),
            "speaker_diarization": isinstance(self.primary_service, WhisperXSTTService),
            "language_detection": hasattr(self.primary_service, 'detect_language'),
            "batch_processing": True,
            "stream_processing": True
        }
