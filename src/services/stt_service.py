"""
Сервис Speech-to-Text (STT) с использованием faster-whisper
Распознает речь в текст с поддержкой разных языков
"""
import asyncio
import numpy as np
from typing import List, Optional, Dict, Any
from faster_whisper import WhisperModel
from src.models.interview import AudioSegment, SpeechTranscription
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class STTService:
    """
    Сервис распознавания речи на основе faster-whisper
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.STTService")
        self.model = None
        self.model_size = settings.STT_MODEL_SIZE
        self.language = settings.STT_LANGUAGE
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация модели Whisper"""
        try:
            # Загружаем модель faster-whisper
            # compute_type="int8" для ускорения на CPU
            # device="cpu" для работы на CPU
            self.model = WhisperModel(
                model_size_or_path=self.model_size,
                device="cpu",
                compute_type="int8",
                download_root=None
            )
            
            self.logger.info(f"Модель Whisper {self.model_size} успешно загружена")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели Whisper: {e}")
            self.model = None
    
    async def transcribe_audio(
        self, 
        audio_segment: AudioSegment,
        language: Optional[str] = None
    ) -> Optional[SpeechTranscription]:
        """
        Транскрипция аудио сегмента
        
        Args:
            audio_segment: Аудио сегмент для транскрипции
            language: Язык речи (если None, используется автоопределение)
            
        Returns:
            Транскрипция речи или None при ошибке
        """
        try:
            if self.model is None:
                self.logger.error("Модель Whisper не загружена")
                return None
            
            # Конвертируем аудио в numpy array
            audio_array = np.frombuffer(audio_segment.audio_data, dtype=np.float32)
            
            # Проверяем, что аудио не пустое
            if len(audio_array) == 0:
                self.logger.warning("Пустой аудио сегмент")
                return None
            
            # Определяем язык если не указан
            target_language = language or self.language
            
            # Выполняем транскрипцию
            segments, info = self.model.transcribe(
                audio_array,
                language=target_language if target_language != "auto" else None,
                task="transcribe",
                beam_size=5,
                best_of=5,
                temperature=0.0,
                compression_ratio_threshold=2.4,
                log_prob_threshold=-1.0,
                no_speech_threshold=0.6,
                condition_on_previous_text=True,
                initial_prompt=None
            )
            
            # Получаем лучший сегмент
            best_segment = None
            best_confidence = 0.0
            
            for segment in segments:
                if segment.no_speech_prob < 0.6:  # Фильтруем сегменты без речи
                    confidence = 1.0 - segment.no_speech_prob
                    if confidence > best_confidence:
                        best_confidence = confidence
                        best_segment = segment
            
            if best_segment is None:
                self.logger.warning("Не удалось распознать речь в аудио")
                return None
            
            # Создаем объект транскрипции
            transcription = SpeechTranscription(
                text=best_segment.text.strip(),
                confidence=best_confidence,
                language=info.language,
                start_time=audio_segment.start_time,
                end_time=audio_segment.end_time,
                audio_segment=audio_segment
            )
            
            self.logger.info(f"Транскрипция: '{transcription.text}' (уверенность: {best_confidence:.2f})")
            return transcription
            
        except Exception as e:
            self.logger.error(f"Ошибка транскрипции: {e}")
            return None
    
    async def transcribe_batch(
        self, 
        audio_segments: List[AudioSegment],
        language: Optional[str] = None
    ) -> List[SpeechTranscription]:
        """
        Пакетная транскрипция нескольких аудио сегментов
        
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
        Потоковая транскрипция с колбэком
        
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
        return [
            "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", "fo", "fr", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", "te", "tg", "th", "tk", "tl", "tr", "tt", "ug", "uk", "ur", "uz", "vi", "yi", "yo", "zh"
        ]
    
    def detect_language(self, audio_segment: AudioSegment) -> Optional[str]:
        """
        Автоопределение языка речи
        
        Args:
            audio_segment: Аудио сегмент
            
        Returns:
            Код языка или None
        """
        try:
            if self.model is None:
                return None
            
            audio_array = np.frombuffer(audio_segment.audio_data, dtype=np.float32)
            
            # Используем короткий сегмент для определения языка
            if len(audio_array) > 16000:  # Берем первые 1 секунду
                audio_array = audio_array[:16000]
            
            # Определяем язык
            segments, info = self.model.transcribe(
                audio_array,
                language=None,  # Автоопределение
                task="transcribe",
                beam_size=1,
                best_of=1
            )
            
            return info.language
            
        except Exception as e:
            self.logger.error(f"Ошибка определения языка: {e}")
            return None
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_size": self.model_size,
            "device": "cpu",
            "compute_type": "int8",
            "supported_languages": self.get_supported_languages()
        }
    
    async def health_check(self) -> bool:
        """Проверка работоспособности сервиса"""
        try:
            if self.model is None:
                return False
            
            # Создаем тестовый аудио сегмент
            test_audio = np.zeros(1600, dtype=np.float32)  # 0.1 секунды тишины
            test_segment = AudioSegment(
                start_time=0.0,
                end_time=0.1,
                duration=0.1,
                audio_data=test_audio.tobytes(),
                confidence=1.0,
                is_speech=False
            )
            
            # Пытаемся выполнить транскрипцию
            result = await self.transcribe_audio(test_segment)
            
            # Результат может быть None для тишины, но ошибок быть не должно
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка health check: {e}")
            return False
