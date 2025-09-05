"""
Улучшенный STT сервис с использованием WhisperX
Включает диаризацию, выравнивание слов и улучшенную точность
"""
import asyncio
import numpy as np
import tempfile
import os
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from src.models.interview import AudioSegment, SpeechTranscription
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class WordSegment:
    """Сегмент слова с временными метками"""
    word: str
    start: float
    end: float
    confidence: float


@dataclass
class SpeakerSegment:
    """Сегмент речи спикера"""
    speaker: str
    start: float
    end: float
    text: str
    words: List[WordSegment]
    confidence: float


class WhisperXSTTService:
    """
    Улучшенный STT сервис на основе WhisperX
    
    Возможности:
    - Высокоточная транскрипция
    - Диаризация (разделение спикеров)
    - Выравнивание слов
    - Поддержка длинных аудио
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.WhisperXSTTService")
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None
        self.model_size = settings.STT_MODEL_SIZE
        self.language = settings.STT_LANGUAGE
        self._initialize_models()
    
    def _initialize_models(self):
        """Инициализация моделей WhisperX"""
        try:
            import whisperx
            
            # Загружаем основную модель Whisper
            self.model = whisperx.load_model(
                self.model_size, 
                device="cpu",  # Используем CPU для совместимости
                compute_type="int8"
            )
            
            self.logger.info(f"WhisperX модель {self.model_size} загружена")
            
            # Инициализируем модели для выравнивания и диаризации
            self._initialize_alignment_model()
            self._initialize_diarization_model()
            
        except ImportError:
            self.logger.error("WhisperX не установлен. Установите: pip install whisperx")
            self.model = None
        except Exception as e:
            self.logger.error(f"Ошибка загрузки WhisperX: {e}")
            self.model = None
    
    def _initialize_alignment_model(self):
        """Инициализация модели для выравнивания слов"""
        try:
            import whisperx
            
            # Загружаем модель для выравнивания
            self.align_model, self.align_metadata = whisperx.load_align_model(
                language_code=self.language if self.language != "auto" else "ru",
                device="cpu"
            )
            
            self.logger.info("Модель выравнивания слов загружена")
            
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить модель выравнивания: {e}")
            self.align_model = None
    
    def _initialize_diarization_model(self):
        """Инициализация модели для диаризации"""
        try:
            import whisperx
            
            # Загружаем модель для диаризации
            self.diarize_model = whisperx.DiarizationPipeline(
                use_auth_token=None,  # Для публичных моделей
                device="cpu"
            )
            
            self.logger.info("Модель диаризации загружена")
            
        except Exception as e:
            self.logger.warning(f"Не удалось загрузить модель диаризации: {e}")
            self.diarize_model = None
    
    async def transcribe_audio(
        self, 
        audio_segment: AudioSegment,
        language: Optional[str] = None,
        enable_diarization: bool = True,
        enable_word_alignment: bool = True
    ) -> Optional[SpeechTranscription]:
        """
        Транскрипция аудио с использованием WhisperX
        
        Args:
            audio_segment: Аудио сегмент для транскрипции
            language: Язык речи
            enable_diarization: Включить диаризацию
            enable_word_alignment: Включить выравнивание слов
            
        Returns:
            Транскрипция речи или None при ошибке
        """
        try:
            if self.model is None:
                self.logger.error("WhisperX модель не загружена")
                return None
            
            # Конвертируем аудио в numpy array
            audio_array = np.frombuffer(audio_segment.audio_data, dtype=np.float32)
            
            if len(audio_array) == 0:
                self.logger.warning("Пустой аудио сегмент")
                return None
            
            # Создаем временный файл для WhisperX
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                # Сохраняем аудио во временный файл
                import soundfile as sf
                sf.write(temp_path, audio_array, 16000)
                
                # Выполняем транскрипцию
                result = await self._transcribe_with_whisperx(
                    temp_path, 
                    language or self.language,
                    enable_diarization,
                    enable_word_alignment
                )
                
                if result:
                    # Создаем объект транскрипции
                    transcription = SpeechTranscription(
                        text=result['text'],
                        confidence=result['confidence'],
                        language=result['language'],
                        start_time=audio_segment.start_time,
                        end_time=audio_segment.end_time,
                        audio_segment=audio_segment
                    )
                    
                    self.logger.info(f"WhisperX транскрипция: '{transcription.text}' (уверенность: {transcription.confidence:.2f})")
                    return transcription
                else:
                    return None
                    
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            self.logger.error(f"Ошибка WhisperX транскрипции: {e}")
            return None
    
    async def _transcribe_with_whisperx(
        self, 
        audio_path: str,
        language: str,
        enable_diarization: bool,
        enable_word_alignment: bool
    ) -> Optional[Dict[str, Any]]:
        """Внутренняя транскрипция с WhisperX"""
        try:
            import whisperx
            
            # 1. Основная транскрипция
            result = self.model.transcribe(
                audio_path,
                language=language if language != "auto" else None,
                batch_size=16
            )
            
            if not result or 'segments' not in result:
                return None
            
            # 2. Выравнивание слов (если доступно)
            if enable_word_alignment and self.align_model is not None:
                result = whisperx.align(
                    result["segments"], 
                    self.align_model, 
                    self.align_metadata, 
                    audio_path, 
                    "cpu"
                )
            
            # 3. Диаризация (если доступно)
            if enable_diarization and self.diarize_model is not None:
                diarize_segments = self.diarize_model(audio_path)
                result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # 4. Обрабатываем результат
            return self._process_whisperx_result(result, language)
            
        except Exception as e:
            self.logger.error(f"Ошибка внутренней транскрипции: {e}")
            return None
    
    def _process_whisperx_result(
        self, 
        result: Dict[str, Any], 
        language: str
    ) -> Dict[str, Any]:
        """Обработка результата WhisperX"""
        try:
            # Объединяем весь текст
            full_text = ""
            total_confidence = 0.0
            segment_count = 0
            
            for segment in result.get('segments', []):
                if 'text' in segment:
                    full_text += segment['text'] + " "
                    total_confidence += segment.get('avg_logprob', 0.0)
                    segment_count += 1
            
            # Вычисляем среднюю уверенность
            avg_confidence = total_confidence / segment_count if segment_count > 0 else 0.0
            # Конвертируем logprob в вероятность (приблизительно)
            confidence = min(1.0, max(0.0, (avg_confidence + 1.0) / 2.0))
            
            return {
                'text': full_text.strip(),
                'confidence': confidence,
                'language': language,
                'segments': result.get('segments', []),
                'word_segments': self._extract_word_segments(result),
                'speaker_segments': self._extract_speaker_segments(result)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки результата: {e}")
            return None
    
    def _extract_word_segments(self, result: Dict[str, Any]) -> List[WordSegment]:
        """Извлечение сегментов слов"""
        word_segments = []
        
        for segment in result.get('segments', []):
            if 'words' in segment:
                for word_info in segment['words']:
                    word_segments.append(WordSegment(
                        word=word_info.get('word', ''),
                        start=word_info.get('start', 0.0),
                        end=word_info.get('end', 0.0),
                        confidence=word_info.get('score', 0.0)
                    ))
        
        return word_segments
    
    def _extract_speaker_segments(self, result: Dict[str, Any]) -> List[SpeakerSegment]:
        """Извлечение сегментов спикеров"""
        speaker_segments = []
        
        for segment in result.get('segments', []):
            speaker = segment.get('speaker', 'unknown')
            words = []
            
            if 'words' in segment:
                for word_info in segment['words']:
                    words.append(WordSegment(
                        word=word_info.get('word', ''),
                        start=word_info.get('start', 0.0),
                        end=word_info.get('end', 0.0),
                        confidence=word_info.get('score', 0.0)
                    ))
            
            speaker_segments.append(SpeakerSegment(
                speaker=speaker,
                start=segment.get('start', 0.0),
                end=segment.get('end', 0.0),
                text=segment.get('text', ''),
                words=words,
                confidence=segment.get('avg_logprob', 0.0)
            ))
        
        return speaker_segments
    
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
            Результат с разделением спикеров
        """
        try:
            # Создаем временный файл
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                # Сохраняем аудио
                import soundfile as sf
                audio_array = np.frombuffer(audio_segment.audio_data, dtype=np.float32)
                sf.write(temp_path, audio_array, 16000)
                
                # Транскрипция с диаризацией
                result = await self._transcribe_with_whisperx(
                    temp_path,
                    language or self.language,
                    enable_diarization=True,
                    enable_word_alignment=True
                )
                
                return result
                
            finally:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
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
    
    def get_supported_languages(self) -> List[str]:
        """Получение списка поддерживаемых языков"""
        return [
            "af", "am", "ar", "as", "az", "ba", "be", "bg", "bn", "bo", "br", "bs", 
            "ca", "cs", "cy", "da", "de", "el", "en", "es", "et", "eu", "fa", "fi", 
            "fo", "fr", "gl", "gu", "ha", "he", "hi", "hr", "ht", "hu", "hy", "id", 
            "is", "it", "ja", "jw", "ka", "kk", "km", "kn", "ko", "la", "lb", "ln", 
            "lo", "lt", "lv", "mg", "mi", "mk", "ml", "mn", "mr", "ms", "mt", "my", 
            "ne", "nl", "nn", "no", "oc", "pa", "pl", "ps", "pt", "ro", "ru", "sa", 
            "sd", "si", "sk", "sl", "sn", "so", "sq", "sr", "su", "sv", "sw", "ta", 
            "te", "tg", "th", "tk", "tl", "tr", "tt", "ug", "uk", "ur", "uz", "vi", 
            "yi", "yo", "zh"
        ]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели"""
        if self.model is None:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_size": self.model_size,
            "device": "cpu",
            "compute_type": "int8",
            "alignment_available": self.align_model is not None,
            "diarization_available": self.diarize_model is not None,
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
