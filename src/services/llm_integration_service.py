"""
Сервис интеграции с LLM
Подготавливает данные для отправки в LLM и обрабатывает ответы
"""
import asyncio
import json
import tempfile
import os
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
from src.models.interview import AudioSegment, SpeechTranscription, ProsodyFeatures, EmotionAnalysis
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class LLMConfig:
    """Конфигурация для LLM интеграции"""
    enable_file_output: bool = True
    output_directory: str = "llm_data"
    include_audio_metadata: bool = True
    include_prosody_features: bool = True
    include_emotion_analysis: bool = True
    include_speaker_info: bool = True
    max_text_length: int = 10000  # Максимальная длина текста для LLM


@dataclass
class LLMInputData:
    """Данные для отправки в LLM"""
    text_content: str
    metadata: Dict[str, Any]
    audio_info: Dict[str, Any]
    prosody_features: Optional[Dict[str, Any]] = None
    emotion_analysis: Optional[Dict[str, Any]] = None
    speaker_info: Optional[Dict[str, Any]] = None


@dataclass
class LLMOutputData:
    """Ответ от LLM"""
    response_text: str
    confidence: float
    processing_time: float
    metadata: Dict[str, Any]


class LLMIntegrationService:
    """
    Сервис для интеграции с LLM
    
    Функции:
    - Подготовка данных для LLM
    - Форматирование текста и метаданных
    - Сохранение в файлы для внешней обработки
    - Обработка ответов от LLM
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.logger = get_logger(f"{__name__}.LLMIntegrationService")
        self.config = config or LLMConfig()
        
        # Создаем директорию для выходных файлов
        if self.config.enable_file_output:
            os.makedirs(self.config.output_directory, exist_ok=True)
        
        # Статистика
        self.processed_requests = 0
        self.saved_files = 0
        self.total_processing_time = 0.0
        
        self.logger.info("LLM Integration сервис инициализирован")
    
    async def prepare_llm_input(
        self,
        transcriptions: List[SpeechTranscription],
        prosody_features: Optional[List[ProsodyFeatures]] = None,
        emotion_analysis: Optional[List[EmotionAnalysis]] = None,
        speaker_info: Optional[Dict[str, Any]] = None,
        additional_context: Optional[Dict[str, Any]] = None
    ) -> LLMInputData:
        """
        Подготовка данных для отправки в LLM
        
        Args:
            transcriptions: Список транскрипций
            prosody_features: Особенности просодии
            emotion_analysis: Анализ эмоций
            speaker_info: Информация о спикерах
            additional_context: Дополнительный контекст
            
        Returns:
            Подготовленные данные для LLM
        """
        try:
            start_time = asyncio.get_event_loop().time()
            
            # Объединяем все транскрипции в единый текст
            text_content = self._combine_transcriptions(transcriptions)
            
            # Подготавливаем метаданные
            metadata = self._prepare_metadata(
                transcriptions, prosody_features, emotion_analysis, 
                speaker_info, additional_context
            )
            
            # Подготавливаем информацию об аудио
            audio_info = self._prepare_audio_info(transcriptions)
            
            # Подготавливаем особенности просодии
            prosody_data = None
            if self.config.include_prosody_features and prosody_features:
                prosody_data = self._prepare_prosody_data(prosody_features)
            
            # Подготавливаем анализ эмоций
            emotion_data = None
            if self.config.include_emotion_analysis and emotion_analysis:
                emotion_data = self._prepare_emotion_data(emotion_analysis)
            
            # Подготавливаем информацию о спикерах
            speaker_data = None
            if self.config.include_speaker_info and speaker_info:
                speaker_data = speaker_info
            
            # Создаем объект данных для LLM
            llm_input = LLMInputData(
                text_content=text_content,
                metadata=metadata,
                audio_info=audio_info,
                prosody_features=prosody_data,
                emotion_analysis=emotion_data,
                speaker_info=speaker_data
            )
            
            processing_time = asyncio.get_event_loop().time() - start_time
            self.total_processing_time += processing_time
            
            self.logger.info(f"Данные для LLM подготовлены за {processing_time:.3f}s")
            
            return llm_input
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки данных для LLM: {e}")
            raise
    
    def _combine_transcriptions(self, transcriptions: List[SpeechTranscription]) -> str:
        """Объединение транскрипций в единый текст"""
        try:
            if not transcriptions:
                return ""
            
            # Сортируем по времени начала
            sorted_transcriptions = sorted(transcriptions, key=lambda t: t.start_time)
            
            # Объединяем текст
            combined_text = []
            for transcription in sorted_transcriptions:
                if transcription.text and transcription.text.strip():
                    combined_text.append(transcription.text.strip())
            
            result = " ".join(combined_text)
            
            # Ограничиваем длину если нужно
            if len(result) > self.config.max_text_length:
                result = result[:self.config.max_text_length] + "..."
                self.logger.warning(f"Текст обрезан до {self.config.max_text_length} символов")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка объединения транскрипций: {e}")
            return ""
    
    def _prepare_metadata(
        self,
        transcriptions: List[SpeechTranscription],
        prosody_features: Optional[List[ProsodyFeatures]],
        emotion_analysis: Optional[List[EmotionAnalysis]],
        speaker_info: Optional[Dict[str, Any]],
        additional_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Подготовка метаданных"""
        try:
            metadata = {
                "timestamp": asyncio.get_event_loop().time(),
                "transcription_count": len(transcriptions),
                "total_duration": sum(t.audio_segment.duration for t in transcriptions if t.audio_segment),
                "average_confidence": sum(t.confidence for t in transcriptions) / len(transcriptions) if transcriptions else 0.0,
                "languages": list(set(t.language for t in transcriptions if t.language)),
                "has_prosody": prosody_features is not None and len(prosody_features) > 0,
                "has_emotion": emotion_analysis is not None and len(emotion_analysis) > 0,
                "has_speaker_info": speaker_info is not None
            }
            
            # Добавляем дополнительный контекст
            if additional_context:
                metadata.update(additional_context)
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки метаданных: {e}")
            return {}
    
    def _prepare_audio_info(self, transcriptions: List[SpeechTranscription]) -> Dict[str, Any]:
        """Подготовка информации об аудио"""
        try:
            if not transcriptions:
                return {}
            
            audio_segments = [t.audio_segment for t in transcriptions if t.audio_segment]
            
            return {
                "segment_count": len(audio_segments),
                "total_duration": sum(seg.duration for seg in audio_segments),
                "start_time": min(seg.start_time for seg in audio_segments) if audio_segments else 0.0,
                "end_time": max(seg.end_time for seg in audio_segments) if audio_segments else 0.0,
                "average_confidence": sum(seg.confidence for seg in audio_segments) / len(audio_segments) if audio_segments else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки аудио информации: {e}")
            return {}
    
    def _prepare_prosody_data(self, prosody_features: List[ProsodyFeatures]) -> Dict[str, Any]:
        """Подготовка данных просодии"""
        try:
            if not prosody_features:
                return {}
            
            # Агрегируем особенности просодии
            avg_pitch = sum(f.pitch for f in prosody_features) / len(prosody_features)
            avg_energy = sum(f.energy for f in prosody_features) / len(prosody_features)
            avg_speaking_rate = sum(f.speaking_rate for f in prosody_features) / len(prosody_features)
            avg_pause_duration = sum(f.pause_duration for f in prosody_features) / len(prosody_features)
            
            return {
                "average_pitch": avg_pitch,
                "average_energy": avg_energy,
                "average_speaking_rate": avg_speaking_rate,
                "average_pause_duration": avg_pause_duration,
                "feature_count": len(prosody_features),
                "voice_quality": prosody_features[0].voice_quality if prosody_features else "unknown"
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки данных просодии: {e}")
            return {}
    
    def _prepare_emotion_data(self, emotion_analysis: List[EmotionAnalysis]) -> Dict[str, Any]:
        """Подготовка данных эмоций"""
        try:
            if not emotion_analysis:
                return {}
            
            # Агрегируем анализ эмоций
            emotions = [e.primary_emotion for e in emotion_analysis]
            emotion_counts = {}
            for emotion in emotions:
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
            
            avg_confidence = sum(e.confidence for e in emotion_analysis) / len(emotion_analysis)
            avg_valence = sum(e.valence for e in emotion_analysis) / len(emotion_analysis)
            avg_arousal = sum(e.arousal for e in emotion_analysis) / len(emotion_analysis)
            
            return {
                "emotion_distribution": emotion_counts,
                "dominant_emotion": max(emotion_counts, key=emotion_counts.get) if emotion_counts else "neutral",
                "average_confidence": avg_confidence,
                "average_valence": avg_valence,
                "average_arousal": avg_arousal,
                "analysis_count": len(emotion_analysis)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка подготовки данных эмоций: {e}")
            return {}
    
    async def save_llm_input_to_file(
        self, 
        llm_input: LLMInputData,
        filename: Optional[str] = None
    ) -> str:
        """
        Сохранение данных для LLM в файл
        
        Args:
            llm_input: Данные для LLM
            filename: Имя файла (если None, генерируется автоматически)
            
        Returns:
            Путь к сохраненному файлу
        """
        try:
            if not self.config.enable_file_output:
                raise ValueError("Файловый вывод отключен")
            
            # Генерируем имя файла если не указано
            if not filename:
                timestamp = int(asyncio.get_event_loop().time() * 1000)
                filename = f"llm_input_{timestamp}.txt"
            
            filepath = os.path.join(self.config.output_directory, filename)
            
            # Подготавливаем содержимое файла
            file_content = self._format_llm_input_for_file(llm_input)
            
            # Сохраняем файл
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(file_content)
            
            self.saved_files += 1
            self.logger.info(f"Данные для LLM сохранены в файл: {filepath}")
            
            return filepath
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения файла: {e}")
            raise
    
    def _format_llm_input_for_file(self, llm_input: LLMInputData) -> str:
        """Форматирование данных для сохранения в файл"""
        try:
            content = []
            
            # Заголовок
            content.append("=== AI HR BOT - LLM INPUT DATA ===")
            content.append(f"Timestamp: {llm_input.metadata.get('timestamp', 'unknown')}")
            content.append("")
            
            # Основной текст
            content.append("=== TRANSCRIPTION TEXT ===")
            content.append(llm_input.text_content)
            content.append("")
            
            # Метаданные
            content.append("=== METADATA ===")
            content.append(json.dumps(llm_input.metadata, ensure_ascii=False, indent=2))
            content.append("")
            
            # Аудио информация
            content.append("=== AUDIO INFO ===")
            content.append(json.dumps(llm_input.audio_info, ensure_ascii=False, indent=2))
            content.append("")
            
            # Особенности просодии
            if llm_input.prosody_features:
                content.append("=== PROSODY FEATURES ===")
                content.append(json.dumps(llm_input.prosody_features, ensure_ascii=False, indent=2))
                content.append("")
            
            # Анализ эмоций
            if llm_input.emotion_analysis:
                content.append("=== EMOTION ANALYSIS ===")
                content.append(json.dumps(llm_input.emotion_analysis, ensure_ascii=False, indent=2))
                content.append("")
            
            # Информация о спикерах
            if llm_input.speaker_info:
                content.append("=== SPEAKER INFO ===")
                content.append(json.dumps(llm_input.speaker_info, ensure_ascii=False, indent=2))
                content.append("")
            
            return "\n".join(content)
            
        except Exception as e:
            self.logger.error(f"Ошибка форматирования файла: {e}")
            return str(llm_input)
    
    async def process_llm_output(
        self, 
        response_text: str,
        processing_time: float,
        metadata: Optional[Dict[str, Any]] = None
    ) -> LLMOutputData:
        """
        Обработка ответа от LLM
        
        Args:
            response_text: Текст ответа от LLM
            processing_time: Время обработки
            metadata: Дополнительные метаданные
            
        Returns:
            Обработанные данные от LLM
        """
        try:
            # Анализируем ответ
            confidence = self._analyze_response_confidence(response_text)
            
            # Подготавливаем метаданные
            output_metadata = metadata or {}
            output_metadata.update({
                "response_length": len(response_text),
                "processing_time": processing_time,
                "timestamp": asyncio.get_event_loop().time()
            })
            
            # Создаем объект ответа
            llm_output = LLMOutputData(
                response_text=response_text,
                confidence=confidence,
                processing_time=processing_time,
                metadata=output_metadata
            )
            
            self.processed_requests += 1
            self.logger.info(f"Ответ от LLM обработан: {len(response_text)} символов, уверенность: {confidence:.2f}")
            
            return llm_output
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки ответа LLM: {e}")
            raise
    
    def _analyze_response_confidence(self, response_text: str) -> float:
        """Анализ уверенности ответа"""
        try:
            if not response_text or not response_text.strip():
                return 0.0
            
            # Простая эвристика для определения уверенности
            confidence = 0.5  # Базовая уверенность
            
            # Увеличиваем уверенность за длину ответа
            if len(response_text) > 50:
                confidence += 0.2
            
            # Увеличиваем уверенность за структурированность
            if any(marker in response_text.lower() for marker in ['ответ:', 'рекомендация:', 'вывод:']):
                confidence += 0.2
            
            # Уменьшаем уверенность за неопределенность
            if any(marker in response_text.lower() for marker in ['не знаю', 'не уверен', 'возможно']):
                confidence -= 0.2
            
            return max(0.0, min(1.0, confidence))
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа уверенности: {e}")
            return 0.5
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики сервиса"""
        return {
            "processed_requests": self.processed_requests,
            "saved_files": self.saved_files,
            "total_processing_time": self.total_processing_time,
            "average_processing_time": self.total_processing_time / self.processed_requests if self.processed_requests > 0 else 0.0
        }
    
    def update_config(self, config: LLMConfig):
        """Обновление конфигурации"""
        self.config = config
        if self.config.enable_file_output:
            os.makedirs(self.config.output_directory, exist_ok=True)
        self.logger.info("Конфигурация LLM Integration обновлена")
    
    def get_config(self) -> LLMConfig:
        """Получение текущей конфигурации"""
        return self.config
    
    async def health_check(self) -> bool:
        """Проверка работоспособности сервиса"""
        try:
            # Проверяем возможность создания файла
            if self.config.enable_file_output:
                test_file = os.path.join(self.config.output_directory, "health_check.txt")
                with open(test_file, 'w') as f:
                    f.write("health check")
                os.remove(test_file)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка health check: {e}")
            return False
