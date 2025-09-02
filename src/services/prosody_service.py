"""
Сервис анализа просодических характеристик речи
Включает анализ интонации, темпа, громкости и пауз
"""
import asyncio
import numpy as np
import librosa
from typing import List, Optional, Dict, Any
from src.models.interview import AudioSegment, ProsodyFeatures
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class ProsodyAnalysisService:
    """
    Сервис анализа просодических характеристик речи
    
    Анализирует:
    - Высоту тона (pitch)
    - Энергию сигнала
    - Темп речи
    - Паузы
    - Качество голоса
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.ProsodyAnalysisService")
        self.sample_rate = 16000
        self.frame_length = int(0.025 * self.sample_rate)  # 25ms
        self.hop_length = int(0.010 * self.sample_rate)    # 10ms
        
        # Инициализация Parselmouth (если доступен)
        self.parselmouth_available = self._check_parselmouth()
    
    def _check_parselmouth(self) -> bool:
        """Проверка доступности Parselmouth"""
        try:
            import parselmouth
            return True
        except ImportError:
            self.logger.warning("Parselmouth не доступен, используется librosa")
            return False
    
    async def analyze_prosody(
        self, 
        audio_segment: AudioSegment
    ) -> Optional[ProsodyFeatures]:
        """
        Анализ просодических характеристик аудио сегмента
        
        Args:
            audio_segment: Аудио сегмент для анализа
            
        Returns:
            Просодические характеристики или None при ошибке
        """
        try:
            # Конвертируем аудио в numpy array
            audio_array = np.frombuffer(audio_segment.audio_data, dtype=np.float32)
            
            if len(audio_array) == 0:
                self.logger.warning("Пустой аудио сегмент для анализа просодии")
                return None
            
            # Анализируем различные характеристики
            pitch_features = await self._analyze_pitch(audio_array)
            energy_features = await self._analyze_energy(audio_array)
            speaking_rate = await self._analyze_speaking_rate(audio_array)
            pause_duration = await self._analyze_pauses(audio_array)
            voice_quality = await self._analyze_voice_quality(audio_array)
            
            # Создаем объект с характеристиками
            prosody_features = ProsodyFeatures(
                pitch_mean=pitch_features.get('mean', 0.0),
                pitch_std=pitch_features.get('std', 0.0),
                energy_mean=energy_features.get('mean', 0.0),
                energy_std=energy_features.get('std', 0.0),
                speaking_rate=speaking_rate,
                pause_duration=pause_duration,
                voice_quality=voice_quality
            )
            
            self.logger.debug(f"Просодия проанализирована: pitch={prosody_features.pitch_mean:.2f}, energy={prosody_features.energy_mean:.2f}")
            return prosody_features
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа просодии: {e}")
            return None
    
    async def _analyze_pitch(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Анализ высоты тона (pitch)
        
        Args:
            audio: Аудио данные
            
        Returns:
            Словарь с характеристиками pitch
        """
        try:
            if self.parselmouth_available:
                return await self._analyze_pitch_parselmouth(audio)
            else:
                return await self._analyze_pitch_librosa(audio)
                
        except Exception as e:
            self.logger.error(f"Ошибка анализа pitch: {e}")
            return {'mean': 0.0, 'std': 0.0}
    
    async def _analyze_pitch_parselmouth(self, audio: np.ndarray) -> Dict[str, float]:
        """Анализ pitch с использованием Parselmouth"""
        try:
            import parselmouth
            
            # Создаем временный файл для Parselmouth
            import tempfile
            import soundfile as sf
            
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                sf.write(tmp_file.name, audio, self.sample_rate)
                
                # Анализируем с Parselmouth
                sound = parselmouth.Sound(tmp_file.name)
                pitch = sound.to_pitch()
                
                # Получаем значения pitch
                pitch_values = pitch.selected_array['frequency']
                pitch_values = pitch_values[pitch_values > 0]  # Убираем невалидные значения
                
                if len(pitch_values) > 0:
                    return {
                        'mean': float(np.mean(pitch_values)),
                        'std': float(np.std(pitch_values))
                    }
                else:
                    return {'mean': 0.0, 'std': 0.0}
                    
        except Exception as e:
            self.logger.error(f"Ошибка Parselmouth pitch анализа: {e}")
            return await self._analyze_pitch_librosa(audio)
    
    async def _analyze_pitch_librosa(self, audio: np.ndarray) -> Dict[str, float]:
        """Анализ pitch с использованием librosa"""
        try:
            # Используем librosa для анализа pitch
            pitches, magnitudes = librosa.piptrack(
                y=audio, 
                sr=self.sample_rate,
                hop_length=self.hop_length,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            
            # Получаем доминирующие частоты
            dominant_pitches = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    dominant_pitches.append(pitch)
            
            if dominant_pitches:
                return {
                    'mean': float(np.mean(dominant_pitches)),
                    'std': float(np.std(dominant_pitches))
                }
            else:
                return {'mean': 0.0, 'std': 0.0}
                
        except Exception as e:
            self.logger.error(f"Ошибка librosa pitch анализа: {e}")
            return {'mean': 0.0, 'std': 0.0}
    
    async def _analyze_energy(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Анализ энергии сигнала
        
        Args:
            audio: Аудио данные
            
        Returns:
            Словарь с характеристиками энергии
        """
        try:
            # Вычисляем энергию по кадрам
            energy = []
            for i in range(0, len(audio) - self.frame_length, self.hop_length):
                frame = audio[i:i + self.frame_length]
                frame_energy = np.sum(frame ** 2)
                energy.append(frame_energy)
            
            if energy:
                energy = np.array(energy)
                return {
                    'mean': float(np.mean(energy)),
                    'std': float(np.std(energy))
                }
            else:
                return {'mean': 0.0, 'std': 0.0}
                
        except Exception as e:
            self.logger.error(f"Ошибка анализа энергии: {e}")
            return {'mean': 0.0, 'std': 0.0}
    
    async def _analyze_speaking_rate(self, audio: np.ndarray) -> float:
        """
        Анализ темпа речи
        
        Args:
            audio: Аудио данные
            
        Returns:
            Темп речи (слоги в секунду)
        """
        try:
            # Простая оценка темпа речи по энергии
            # В реальной реализации лучше использовать более сложные алгоритмы
            
            # Находим пики энергии (потенциальные слоги)
            energy = []
            for i in range(0, len(audio) - self.frame_length, self.hop_length):
                frame = audio[i:i + self.frame_length]
                frame_energy = np.sum(frame ** 2)
                energy.append(frame_energy)
            
            if not energy:
                return 0.0
            
            energy = np.array(energy)
            
            # Находим пики выше среднего
            threshold = np.mean(energy) + 0.5 * np.std(energy)
            peaks = energy > threshold
            
            # Подсчитываем количество пиков
            syllable_count = np.sum(peaks)
            
            # Оцениваем темп речи
            duration = len(audio) / self.sample_rate
            speaking_rate = syllable_count / duration if duration > 0 else 0.0
            
            # Нормализуем к разумному диапазону (2-8 слогов в секунду)
            speaking_rate = max(2.0, min(8.0, speaking_rate))
            
            return float(speaking_rate)
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа темпа речи: {e}")
            return 4.0  # Средний темп по умолчанию
    
    async def _analyze_pauses(self, audio: np.ndarray) -> float:
        """
        Анализ пауз в речи
        
        Args:
            audio: Аудио данные
            
        Returns:
            Длительность пауз в секундах
        """
        try:
            # Анализируем паузы по энергии
            energy = []
            for i in range(0, len(audio) - self.frame_length, self.hop_length):
                frame = audio[i:i + self.frame_length]
                frame_energy = np.sum(frame ** 2)
                energy.append(frame_energy)
            
            if not energy:
                return 0.0
            
            energy = np.array(energy)
            
            # Определяем порог для пауз
            threshold = np.mean(energy) + 0.2 * np.std(energy)
            
            # Находим сегменты с низкой энергией (паузы)
            pause_frames = energy < threshold
            
            # Группируем последовательные паузы
            pause_duration = 0.0
            current_pause_start = None
            
            for i, is_pause in enumerate(pause_frames):
                if is_pause and current_pause_start is None:
                    current_pause_start = i
                elif not is_pause and current_pause_start is not None:
                    # Конец паузы
                    pause_length = (i - current_pause_start) * self.hop_length / self.sample_rate
                    
                    # Учитываем только паузы длиннее 0.1 секунды
                    if pause_length > 0.1:
                        pause_duration += pause_length
                    
                    current_pause_start = None
            
            # Обрабатываем последнюю паузу
            if current_pause_start is not None:
                pause_length = (len(pause_frames) - current_pause_start) * self.hop_length / self.sample_rate
                if pause_length > 0.1:
                    pause_duration += pause_length
            
            return float(pause_duration)
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа пауз: {e}")
            return 0.0
    
    async def _analyze_voice_quality(self, audio: np.ndarray) -> str:
        """
        Анализ качества голоса
        
        Args:
            audio: Аудио данные
            
        Returns:
            Описание качества голоса
        """
        try:
            # Простой анализ качества по SNR и стабильности
            energy = []
            for i in range(0, len(audio) - self.frame_length, self.hop_length):
                frame = audio[i:i + self.frame_length]
                frame_energy = np.sum(frame ** 2)
                energy.append(frame_energy)
            
            if not energy:
                return "unknown"
            
            energy = np.array(energy)
            
            # Анализируем стабильность энергии
            energy_cv = np.std(energy) / np.mean(energy) if np.mean(energy) > 0 else 0
            
            # Анализируем общую энергию
            total_energy = np.mean(energy)
            
            # Определяем качество
            if total_energy > 0.1 and energy_cv < 0.5:
                return "excellent"
            elif total_energy > 0.05 and energy_cv < 1.0:
                return "good"
            elif total_energy > 0.02:
                return "fair"
            else:
                return "poor"
                
        except Exception as e:
            self.logger.error(f"Ошибка анализа качества голоса: {e}")
            return "unknown"
    
    async def analyze_batch(
        self, 
        audio_segments: List[AudioSegment]
    ) -> List[ProsodyFeatures]:
        """
        Пакетный анализ нескольких аудио сегментов
        
        Args:
            audio_segments: Список аудио сегментов
            
        Returns:
            Список характеристик просодии
        """
        features = []
        
        for segment in audio_segments:
            feature = await self.analyze_prosody(segment)
            if feature:
                features.append(feature)
        
        return features
    
    async def health_check(self) -> bool:
        """Проверка работоспособности сервиса"""
        try:
            # Создаем тестовый аудио сегмент
            test_audio = np.random.randn(1600).astype(np.float32) * 0.1
            test_segment = AudioSegment(
                start_time=0.0,
                end_time=0.1,
                duration=0.1,
                audio_data=test_audio.tobytes(),
                confidence=1.0,
                is_speech=True
            )
            
            # Пытаемся проанализировать
            result = await self.analyze_prosody(test_segment)
            
            return result is not None
            
        except Exception as e:
            self.logger.error(f"Ошибка health check: {e}")
            return False
