"""
Сервис Voice Activity Detection (VAD) с использованием Silero
Обнаруживает голосовую активность и сегментирует речь на фразы
"""
import asyncio
import numpy as np
import torch
import torchaudio
from typing import List, Optional, Callable, Tuple
from dataclasses import dataclass
from src.models.interview import AudioSegment
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class VADConfig:
    """Конфигурация VAD"""
    threshold: float = 0.5
    min_speech_duration: float = 0.5
    max_speech_duration: float = 30.0
    min_silence_duration: float = 0.5
    window_size: float = 0.1  # Размер окна в секундах
    step_size: float = 0.05   # Шаг окна в секундах


class SileroVADService:
    """
    Сервис VAD на основе Silero
    """
    
    def __init__(self, config: VADConfig = None):
        self.config = config or VADConfig(
            threshold=settings.VAD_THRESHOLD,
            min_speech_duration=settings.VAD_MIN_SPEECH_DURATION,
            max_speech_duration=settings.VAD_MAX_SPEECH_DURATION
        )
        self.logger = get_logger(f"{__name__}.SileroVADService")
        
        # Инициализация модели Silero
        self.model = None
        self.sample_rate = 16000
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация модели Silero VAD"""
        try:
            # Загружаем предобученную модель Silero VAD
            self.model, utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False
            )
            
            # Получаем функции для работы с моделью
            self.get_speech_timestamps, self.save_audio, self.read_audio, self.VADIterator, self.collect_chunks = utils
            
            self.logger.info("Модель Silero VAD успешно загружена")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели Silero VAD: {e}")
            # Fallback на простой энергетический VAD
            self.model = None
    
    async def detect_speech(
        self, 
        audio_segment: AudioSegment,
        callback: Optional[Callable[[AudioSegment], None]] = None
    ) -> List[AudioSegment]:
        """
        Обнаружение речи в аудио сегменте
        
        Args:
            audio_segment: Аудио сегмент для анализа
            callback: Колбэк для обработки обнаруженных сегментов речи
            
        Returns:
            Список сегментов с речью
        """
        try:
            if self.model is None:
                # Fallback на простой VAD
                return await self._simple_vad(audio_segment, callback)
            
            # Конвертируем аудио в numpy array
            audio_array = np.frombuffer(audio_segment.audio_data, dtype=np.float32)
            
            # Конвертируем в torch tensor
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            
            # Получаем временные метки речи
            speech_timestamps = self.get_speech_timestamps(
                audio_tensor,
                sampling_rate=self.sample_rate,
                threshold=self.config.threshold,
                min_speech_duration_ms=int(self.config.min_speech_duration * 1000),
                max_speech_duration_ms=int(self.config.max_speech_duration * 1000),
                min_silence_duration_ms=int(self.config.min_silence_duration * 1000),
                window_size_ms=int(self.config.window_size * 1000),
                step_size_ms=int(self.config.step_size * 1000)
            )
            
            # Создаем сегменты речи
            speech_segments = []
            for start_ms, end_ms in speech_timestamps:
                start_time = start_ms / 1000.0
                end_time = end_ms / 1000.0
                
                # Извлекаем аудио для этого сегмента
                start_sample = int(start_time * self.sample_rate)
                end_sample = int(end_time * self.sample_rate)
                
                if start_sample < len(audio_array) and end_sample <= len(audio_array):
                    segment_audio = audio_array[start_sample:end_sample]
                    
                    segment = AudioSegment(
                        start_time=audio_segment.start_time + start_time,
                        end_time=audio_segment.start_time + end_time,
                        duration=end_time - start_time,
                        audio_data=segment_audio.tobytes(),
                        confidence=0.9,  # Высокая уверенность для Silero
                        is_speech=True
                    )
                    
                    speech_segments.append(segment)
                    
                    if callback:
                        await callback(segment)
            
            self.logger.info(f"Обнаружено {len(speech_segments)} сегментов речи")
            return speech_segments
            
        except Exception as e:
            self.logger.error(f"Ошибка обнаружения речи: {e}")
            # Fallback на простой VAD
            return await self._simple_vad(audio_segment, callback)
    
    async def _simple_vad(
        self, 
        audio_segment: AudioSegment,
        callback: Optional[Callable[[AudioSegment], None]] = None
    ) -> List[AudioSegment]:
        """
        Простой энергетический VAD как fallback
        
        Args:
            audio_segment: Аудио сегмент
            callback: Колбэк для обработки
            
        Returns:
            Список сегментов с речью
        """
        try:
            audio_array = np.frombuffer(audio_segment.audio_data, dtype=np.float32)
            
            # Вычисляем энергию сигнала
            frame_length = int(0.025 * self.sample_rate)  # 25ms окна
            hop_length = int(0.010 * self.sample_rate)    # 10ms шаг
            
            energy = []
            for i in range(0, len(audio_array) - frame_length, hop_length):
                frame = audio_array[i:i + frame_length]
                frame_energy = np.sum(frame ** 2)
                energy.append(frame_energy)
            
            if not energy:
                return []
            
            energy = np.array(energy)
            
            # Адаптивный порог
            threshold = np.mean(energy) + 0.5 * np.std(energy)
            
            # Находим сегменты выше порога
            speech_frames = energy > threshold
            
            # Группируем последовательные кадры
            segments = []
            start_frame = None
            
            for i, is_speech in enumerate(speech_frames):
                if is_speech and start_frame is None:
                    start_frame = i
                elif not is_speech and start_frame is not None:
                    # Конец сегмента речи
                    end_frame = i
                    duration_frames = end_frame - start_frame
                    
                    if duration_frames * hop_length / self.sample_rate >= self.config.min_speech_duration:
                        start_time = start_frame * hop_length / self.sample_rate
                        end_time = end_frame * hop_length / self.sample_rate
                        
                        start_sample = int(start_time * self.sample_rate)
                        end_sample = int(end_time * self.sample_rate)
                        
                        if start_sample < len(audio_array) and end_sample <= len(audio_array):
                            segment_audio = audio_array[start_sample:end_sample]
                            
                            segment = AudioSegment(
                                start_time=audio_segment.start_time + start_time,
                                end_time=audio_segment.start_time + end_time,
                                duration=end_time - start_time,
                                audio_data=segment_audio.tobytes(),
                                confidence=0.7,  # Средняя уверенность для простого VAD
                                is_speech=True
                            )
                            
                            segments.append(segment)
                            
                            if callback:
                                await callback(segment)
                    
                    start_frame = None
            
            # Обрабатываем последний сегмент
            if start_frame is not None:
                end_frame = len(speech_frames)
                duration_frames = end_frame - start_frame
                
                if duration_frames * hop_length / self.sample_rate >= self.config.min_speech_duration:
                    start_time = start_frame * hop_length / self.sample_rate
                    end_time = end_frame * hop_length / self.sample_rate
                    
                    start_sample = int(start_time * self.sample_rate)
                    end_sample = int(end_time * self.sample_rate)
                    
                    if start_sample < len(audio_array) and end_sample <= len(audio_array):
                        segment_audio = audio_array[start_sample:end_sample]
                        
                        segment = AudioSegment(
                            start_time=audio_segment.start_time + start_time,
                            end_time=audio_segment.start_time + end_time,
                            duration=end_time - start_time,
                            audio_data=segment_audio.tobytes(),
                            confidence=0.7,
                            is_speech=True
                        )
                        
                        segments.append(segment)
                        
                        if callback:
                            await callback(segment)
            
            self.logger.info(f"Простой VAD обнаружил {len(segments)} сегментов речи")
            return segments
            
        except Exception as e:
            self.logger.error(f"Ошибка простого VAD: {e}")
            return []
    
    async def process_audio_stream(
        self, 
        audio_stream: List[AudioSegment],
        callback: Optional[Callable[[AudioSegment], None]] = None
    ) -> List[AudioSegment]:
        """
        Обработка потока аудио сегментов
        
        Args:
            audio_stream: Поток аудио сегментов
            callback: Колбэк для обработки
            
        Returns:
            Список сегментов с речью
        """
        all_speech_segments = []
        
        for segment in audio_stream:
            speech_segments = await self.detect_speech(segment, callback)
            all_speech_segments.extend(speech_segments)
        
        return all_speech_segments
    
    def is_speech(self, audio_data: bytes) -> Tuple[bool, float]:
        """
        Быстрая проверка наличия речи
        
        Args:
            audio_data: Аудио данные
            
        Returns:
            (is_speech, confidence)
        """
        try:
            if self.model is None:
                return self._simple_speech_check(audio_data)
            
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            audio_tensor = torch.from_numpy(audio_array).unsqueeze(0)
            
            # Получаем вероятность речи
            speech_prob = self.model(audio_tensor, self.sample_rate).item()
            
            is_speech = speech_prob > self.config.threshold
            return is_speech, speech_prob
            
        except Exception as e:
            self.logger.error(f"Ошибка проверки речи: {e}")
            return self._simple_speech_check(audio_data)
    
    def _simple_speech_check(self, audio_data: bytes) -> Tuple[bool, float]:
        """Простая проверка речи по энергии"""
        try:
            audio_array = np.frombuffer(audio_data, dtype=np.float32)
            energy = np.sum(audio_array ** 2)
            
            # Простой порог
            threshold = 0.01
            is_speech = energy > threshold
            
            # Нормализованная уверенность
            confidence = min(energy / threshold, 1.0)
            
            return is_speech, confidence
            
        except Exception as e:
            self.logger.error(f"Ошибка простой проверки речи: {e}")
            return False, 0.0
