"""
Сервис Endpointing - определение конца речи
Использует VAD и временные метки для определения завершения высказывания
"""
import asyncio
import time
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from src.models.interview import AudioSegment
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class EndpointConfig:
    """Конфигурация для endpointing"""
    silence_threshold: float = 0.5  # Секунды тишины для определения конца
    min_speech_duration: float = 0.3  # Минимальная длительность речи
    max_speech_duration: float = 30.0  # Максимальная длительность речи
    vad_confidence_threshold: float = 0.5  # Порог уверенности VAD


@dataclass
class SpeechEndpoint:
    """Точка окончания речи"""
    start_time: float
    end_time: float
    duration: float
    confidence: float
    is_complete: bool  # True если речь завершена, False если прервана


class EndpointingService:
    """
    Сервис для определения конца речи
    
    Использует:
    - VAD для обнаружения речи/тишины
    - Временные метки для определения длительности
    - Конфигурационные параметры для настройки
    """
    
    def __init__(self, config: Optional[EndpointConfig] = None):
        self.logger = get_logger(f"{__name__}.EndpointingService")
        self.config = config or EndpointConfig()
        
        # Состояние endpointing
        self.current_speech_start = None
        self.last_speech_time = None
        self.silence_start = None
        self.speech_segments = []
        
        self.logger.info("Endpointing сервис инициализирован")
    
    async def process_audio_segment(
        self, 
        audio_segment: AudioSegment
    ) -> Optional[SpeechEndpoint]:
        """
        Обработка аудио сегмента для определения конца речи
        
        Args:
            audio_segment: Аудио сегмент для анализа
            
        Returns:
            SpeechEndpoint если речь завершена, иначе None
        """
        try:
            current_time = audio_segment.start_time
            is_speech = audio_segment.is_speech
            confidence = audio_segment.confidence
            
            # Обновляем состояние
            if is_speech and confidence >= self.config.vad_confidence_threshold:
                await self._handle_speech_segment(current_time, confidence)
            else:
                await self._handle_silence_segment(current_time)
            
            # Проверяем, завершена ли речь
            endpoint = await self._check_speech_endpoint(current_time)
            
            if endpoint:
                self.logger.debug(f"Обнаружен конец речи: {endpoint.duration:.2f}s")
                return endpoint
            
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки аудио сегмента: {e}")
            return None
    
    async def _handle_speech_segment(self, timestamp: float, confidence: float):
        """Обработка сегмента с речью"""
        try:
            # Если это начало новой речи
            if self.current_speech_start is None:
                self.current_speech_start = timestamp
                self.logger.debug(f"Начало речи: {timestamp:.2f}s")
            
            # Обновляем время последней речи
            self.last_speech_time = timestamp
            self.silence_start = None
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки сегмента речи: {e}")
    
    async def _handle_silence_segment(self, timestamp: float):
        """Обработка сегмента тишины"""
        try:
            # Если мы в процессе речи и началась тишина
            if self.current_speech_start is not None and self.silence_start is None:
                self.silence_start = timestamp
                self.logger.debug(f"Начало тишины: {timestamp:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки сегмента тишины: {e}")
    
    async def _check_speech_endpoint(self, current_time: float) -> Optional[SpeechEndpoint]:
        """Проверка, завершена ли речь"""
        try:
            # Если нет активной речи
            if self.current_speech_start is None:
                return None
            
            # Проверяем максимальную длительность речи
            speech_duration = current_time - self.current_speech_start
            if speech_duration >= self.config.max_speech_duration:
                self.logger.debug("Речь завершена по максимальной длительности")
                return await self._create_endpoint(True)
            
            # Проверяем тишину
            if self.silence_start is not None:
                silence_duration = current_time - self.silence_start
                
                if silence_duration >= self.config.silence_threshold:
                    # Проверяем минимальную длительность речи
                    if speech_duration >= self.config.min_speech_duration:
                        self.logger.debug("Речь завершена по тишине")
                        return await self._create_endpoint(True)
                    else:
                        self.logger.debug("Речь слишком короткая, игнорируем")
                        return await self._create_endpoint(False)
            
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка проверки endpoint: {e}")
            return None
    
    async def _create_endpoint(self, is_complete: bool) -> SpeechEndpoint:
        """Создание точки окончания речи"""
        try:
            if self.current_speech_start is None:
                return None
            
            end_time = self.last_speech_time or self.current_speech_start
            duration = end_time - self.current_speech_start
            
            endpoint = SpeechEndpoint(
                start_time=self.current_speech_start,
                end_time=end_time,
                duration=duration,
                confidence=0.8,  # Базовая уверенность
                is_complete=is_complete
            )
            
            # Сбрасываем состояние
            self._reset_state()
            
            return endpoint
            
        except Exception as e:
            self.logger.error(f"Ошибка создания endpoint: {e}")
            return None
    
    def _reset_state(self):
        """Сброс состояния endpointing"""
        self.current_speech_start = None
        self.last_speech_time = None
        self.silence_start = None
    
    async def process_audio_stream(
        self, 
        audio_segments: List[AudioSegment]
    ) -> List[SpeechEndpoint]:
        """
        Обработка потока аудио сегментов
        
        Args:
            audio_segments: Список аудио сегментов
            
        Returns:
            Список точек окончания речи
        """
        endpoints = []
        
        for segment in audio_segments:
            endpoint = await self.process_audio_segment(segment)
            if endpoint:
                endpoints.append(endpoint)
        
        return endpoints
    
    async def force_endpoint(self) -> Optional[SpeechEndpoint]:
        """
        Принудительное завершение текущей речи
        
        Returns:
            SpeechEndpoint если была активная речь
        """
        try:
            if self.current_speech_start is not None:
                self.logger.debug("Принудительное завершение речи")
                return await self._create_endpoint(True)
            return None
            
        except Exception as e:
            self.logger.error(f"Ошибка принудительного завершения: {e}")
            return None
    
    def get_current_speech_duration(self) -> float:
        """Получение длительности текущей речи"""
        if self.current_speech_start is None:
            return 0.0
        
        current_time = time.time()
        return current_time - self.current_speech_start
    
    def is_speech_active(self) -> bool:
        """Проверка, активна ли речь"""
        return self.current_speech_start is not None
    
    def get_silence_duration(self) -> float:
        """Получение длительности текущей тишины"""
        if self.silence_start is None:
            return 0.0
        
        current_time = time.time()
        return current_time - self.silence_start
    
    def update_config(self, config: EndpointConfig):
        """Обновление конфигурации"""
        self.config = config
        self.logger.info("Конфигурация endpointing обновлена")
    
    def get_config(self) -> EndpointConfig:
        """Получение текущей конфигурации"""
        return self.config
    
    async def health_check(self) -> bool:
        """Проверка работоспособности сервиса"""
        try:
            # Простая проверка - создаем тестовый сегмент
            test_segment = AudioSegment(
                start_time=0.0,
                end_time=0.1,
                duration=0.1,
                audio_data=b'\x00' * 100,
                confidence=0.8,
                is_speech=True
            )
            
            # Пытаемся обработать
            result = await self.process_audio_segment(test_segment)
            
            # Результат может быть None, но ошибок быть не должно
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка health check: {e}")
            return False
