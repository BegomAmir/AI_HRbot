"""
Сервис для обработки аудио потоков
Включает RTP декодирование, ресемплирование и VAD (Voice Activity Detection)
"""
import asyncio
import numpy as np
import librosa
import soundfile as sf
from typing import List, Optional, Tuple, Callable
from dataclasses import dataclass
from src.models.interview import AudioSegment
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class AudioConfig:
    """Конфигурация аудио"""
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    format: str = "wav"


class AudioProcessor:
    """
    Основной класс для обработки аудио потоков
    
    Обрабатывает:
    - RTP декодирование
    - Ресемплирование
    - Нормализацию
    - Сегментацию на чанки
    """
    
    def __init__(self, config: AudioConfig = None):
        self.config = config or AudioConfig(
            sample_rate=settings.AUDIO_SAMPLE_RATE,
            channels=settings.AUDIO_CHANNELS,
            chunk_size=settings.AUDIO_CHUNK_SIZE,
            format=settings.AUDIO_FORMAT
        )
        self.logger = get_logger(f"{__name__}.AudioProcessor")
        
    async def process_rtp_stream(
        self, 
        rtp_data: bytes,
        callback: Optional[Callable[[AudioSegment], None]] = None
    ) -> AudioSegment:
        """
        Обработка RTP потока
        
        Args:
            rtp_data: Сырые RTP данные
            callback: Колбэк для обработки сегментов
            
        Returns:
            Обработанный аудио сегмент
        """
        try:
            # Декодируем RTP заголовок (упрощенная версия)
            audio_data = self._extract_audio_from_rtp(rtp_data)
            
            # Конвертируем в numpy array
            audio_array = self._bytes_to_numpy(audio_data)
            
            # Ресемплируем если нужно
            if self.config.sample_rate != 8000:  # Предполагаем, что RTP использует 8kHz
                audio_array = await self._resample_audio(audio_array, 8000, self.config.sample_rate)
            
            # Нормализуем аудио
            audio_array = self._normalize_audio(audio_array)
            
            # Создаем аудио сегмент
            segment = AudioSegment(
                start_time=0.0,  # Будет обновлено в VAD
                end_time=len(audio_array) / self.config.sample_rate,
                duration=len(audio_array) / self.config.sample_rate,
                audio_data=audio_array.tobytes(),
                confidence=1.0,
                is_speech=True
            )
            
            if callback:
                await callback(segment)
                
            return segment
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки RTP потока: {e}")
            raise
    
    def _extract_audio_from_rtp(self, rtp_data: bytes) -> bytes:
        """
        Извлечение аудио данных из RTP пакета
        
        Args:
            rtp_data: Сырые RTP данные
            
        Returns:
            Аудио данные без RTP заголовка
        """
        # RTP заголовок обычно 12 байт
        # В реальной реализации нужно парсить RTP заголовок
        # Здесь упрощенная версия для демонстрации
        if len(rtp_data) > 12:
            return rtp_data[12:]
        return rtp_data
    
    def _bytes_to_numpy(self, audio_bytes: bytes) -> np.ndarray:
        """
        Конвертация байтов в numpy array
        
        Args:
            audio_bytes: Аудио данные в байтах
            
        Returns:
            Numpy array с аудио данными
        """
        # Предполагаем 16-bit PCM
        audio_array = np.frombuffer(audio_bytes, dtype=np.int16)
        # Конвертируем в float32 для дальнейшей обработки
        return audio_array.astype(np.float32) / 32768.0
    
    async def _resample_audio(
        self, 
        audio: np.ndarray, 
        original_sr: int, 
        target_sr: int
    ) -> np.ndarray:
        """
        Асинхронное ресемплирование аудио
        
        Args:
            audio: Аудио данные
            original_sr: Исходная частота дискретизации
            target_sr: Целевая частота дискретизации
            
        Returns:
            Ресемплированное аудио
        """
        try:
            # Используем librosa для ресемплирования
            # В продакшене лучше использовать более быстрые библиотеки
            resampled = librosa.resample(
                audio, 
                orig_sr=original_sr, 
                target_sr=target_sr
            )
            return resampled
        except Exception as e:
            self.logger.error(f"Ошибка ресемплирования: {e}")
            return audio
    
    def _normalize_audio(self, audio: np.ndarray) -> np.ndarray:
        """
        Нормализация аудио
        
        Args:
            audio: Аудио данные
            
        Returns:
            Нормализованное аудио
        """
        # Убираем DC компонент
        audio = audio - np.mean(audio)
        
        # Нормализуем по амплитуде
        if np.max(np.abs(audio)) > 0:
            audio = audio / np.max(np.abs(audio)) * 0.95
            
        return audio
    
    def segment_audio(
        self, 
        audio: np.ndarray, 
        segment_duration: float = 1.0
    ) -> List[AudioSegment]:
        """
        Сегментация аудио на чанки
        
        Args:
            audio: Аудио данные
            segment_duration: Длительность сегмента в секундах
            
        Returns:
            Список аудио сегментов
        """
        segments = []
        samples_per_segment = int(segment_duration * self.config.sample_rate)
        
        for i in range(0, len(audio), samples_per_segment):
            segment_audio = audio[i:i + samples_per_segment]
            
            if len(segment_audio) > 0:
                segment = AudioSegment(
                    start_time=i / self.config.sample_rate,
                    end_time=(i + len(segment_audio)) / self.config.sample_rate,
                    duration=len(segment_audio) / self.config.sample_rate,
                    audio_data=segment_audio.tobytes(),
                    confidence=1.0,
                    is_speech=True
                )
                segments.append(segment)
        
        return segments
    
    async def process_audio_chunk(
        self, 
        chunk: bytes,
        timestamp: float
    ) -> AudioSegment:
        """
        Обработка одного чанка аудио
        
        Args:
            chunk: Аудио чанк
            timestamp: Временная метка
            
        Returns:
            Обработанный аудио сегмент
        """
        try:
            # Конвертируем в numpy
            audio_array = self._bytes_to_numpy(chunk)
            
            # Нормализуем
            audio_array = self._normalize_audio(audio_array)
            
            # Создаем сегмент
            segment = AudioSegment(
                start_time=timestamp,
                end_time=timestamp + len(audio_array) / self.config.sample_rate,
                duration=len(audio_array) / self.config.sample_rate,
                audio_data=audio_array.tobytes(),
                confidence=1.0,
                is_speech=True
            )
            
            return segment
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки аудио чанка: {e}")
            raise


class AudioBuffer:
    """
    Буфер для накопления аудио данных
    """
    
    def __init__(self, max_duration: float = 5.0):
        self.max_duration = max_duration
        self.buffer: List[AudioSegment] = []
        self.logger = get_logger(f"{__name__}.AudioBuffer")
    
    def add_segment(self, segment: AudioSegment):
        """Добавление сегмента в буфер"""
        self.buffer.append(segment)
        self._cleanup_old_segments()
    
    def _cleanup_old_segments(self):
        """Удаление старых сегментов"""
        if not self.buffer:
            return
            
        current_time = self.buffer[-1].end_time
        cutoff_time = current_time - self.max_duration
        
        # Удаляем сегменты старше cutoff_time
        self.buffer = [s for s in self.buffer if s.end_time > cutoff_time]
    
    def get_buffer_duration(self) -> float:
        """Получение общей длительности буфера"""
        if not self.buffer:
            return 0.0
        return self.buffer[-1].end_time - self.buffer[0].start_time
    
    def get_audio_data(self) -> bytes:
        """Получение всех аудио данных из буфера"""
        if not self.buffer:
            return b""
        
        # Объединяем все сегменты
        all_audio = []
        for segment in self.buffer:
            audio_array = np.frombuffer(segment.audio_data, dtype=np.float32)
            all_audio.append(audio_array)
        
        if all_audio:
            combined = np.concatenate(all_audio)
            return combined.tobytes()
        
        return b""
    
    def clear(self):
        """Очистка буфера"""
        self.buffer.clear()
