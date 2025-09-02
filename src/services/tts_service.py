"""
Сервис Text-to-Speech (TTS) с использованием Piper
Синтезирует речь из текста для ответов бота
"""
import asyncio
import numpy as np
import soundfile as sf
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
import tempfile
import os
from src.models.interview import AudioSegment
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class TTSService:
    """
    Сервис синтеза речи на основе Piper
    
    Поддерживает:
    - Различные голоса и языки
    - Настройка скорости и качества
    - Экспорт в различные аудио форматы
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.TTSService")
        self.voice = settings.TTS_VOICE
        self.speed = settings.TTS_SPEED
        self.sample_rate = 22050  # Стандартная частота для Piper
        
        # Инициализация Piper
        self.piper_model = None
        self.piper_voice = None
        self._initialize_piper()
    
    def _initialize_piper(self):
        """Инициализация Piper TTS"""
        try:
            # Пытаемся импортировать Piper
            import piper
            self.piper_available = True
            
            # Инициализируем Piper
            self.piper_model = piper.PiperVoice.load(
                model_path=None,  # Будет загружено автоматически
                config_path=None,
                use_cuda=False  # Для CPU
            )
            
            self.logger.info("Piper TTS успешно инициализирован")
            
        except ImportError:
            self.logger.warning("Piper не доступен, используется fallback TTS")
            self.piper_available = False
            self.piper_model = None
    
    async def synthesize_speech(
        self, 
        text: str,
        voice: Optional[str] = None,
        speed: Optional[float] = None
    ) -> Optional[AudioSegment]:
        """
        Синтез речи из текста
        
        Args:
            text: Текст для синтеза
            voice: Голос (если None, используется настройка по умолчанию)
            speed: Скорость речи (если None, используется настройка по умолчанию)
            
        Returns:
            Аудио сегмент с синтезированной речью или None при ошибке
        """
        try:
            if not text.strip():
                self.logger.warning("Пустой текст для синтеза")
                return None
            
            target_voice = voice or self.voice
            target_speed = speed or self.speed
            
            self.logger.info(f"Синтез речи: '{text[:50]}...' голос: {target_voice}")
            
            if self.piper_available and self.piper_model:
                return await self._synthesize_with_piper(text, target_voice, target_speed)
            else:
                return await self._synthesize_fallback(text, target_voice, target_speed)
                
        except Exception as e:
            self.logger.error(f"Ошибка синтеза речи: {e}")
            return None
    
    async def _synthesize_with_piper(
        self, 
        text: str, 
        voice: str, 
        speed: float
    ) -> Optional[AudioSegment]:
        """Синтез с использованием Piper"""
        try:
            # Создаем временный файл для аудио
            with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_file:
                temp_path = tmp_file.name
            
            try:
                # Синтезируем речь
                self.piper_model.synthesize(
                    text=text,
                    output_file=temp_path,
                    speaker=voice,
                    length_scale=1.0 / speed,  # Обратная зависимость
                    noise_scale=0.667,
                    noise_w=0.8
                )
                
                # Читаем сгенерированное аудио
                audio_data, sample_rate = sf.read(temp_path)
                
                # Конвертируем в float32 если нужно
                if audio_data.dtype != np.float32:
                    audio_data = audio_data.astype(np.float32)
                
                # Нормализуем аудио
                if np.max(np.abs(audio_data)) > 0:
                    audio_data = audio_data / np.max(np.abs(audio_data)) * 0.95
                
                # Создаем аудио сегмент
                segment = AudioSegment(
                    start_time=0.0,
                    end_time=len(audio_data) / sample_rate,
                    duration=len(audio_data) / sample_rate,
                    audio_data=audio_data.tobytes(),
                    confidence=0.95,  # Высокая уверенность для TTS
                    is_speech=True
                )
                
                self.logger.debug(f"Piper синтезировал {segment.duration:.2f}s аудио")
                return segment
                
            finally:
                # Удаляем временный файл
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
                    
        except Exception as e:
            self.logger.error(f"Ошибка Piper синтеза: {e}")
            return None
    
    async def _synthesize_fallback(
        self, 
        text: str, 
        voice: str, 
        speed: float
    ) -> Optional[AudioSegment]:
        """Fallback синтез речи (простая синусоида)"""
        try:
            # Простой fallback - генерируем тональный сигнал
            # В реальной реализации здесь можно использовать другие TTS библиотеки
            
            # Длительность аудио зависит от длины текста
            duration = min(max(len(text) * 0.1, 0.5), 10.0)  # 0.5-10 секунд
            
            # Генерируем тональный сигнал
            frequency = 440  # 440 Hz (нота A)
            t = np.linspace(0, duration, int(self.sample_rate * duration), False)
            
            # Создаем основной тон
            audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            # Добавляем модуляцию для имитации речи
            modulation = np.sin(2 * np.pi * 5 * t) * 0.3
            audio_data = audio_data * (1 + modulation)
            
            # Применяем скорость
            if speed != 1.0:
                # Простое изменение длительности
                new_duration = duration / speed
                new_samples = int(self.sample_rate * new_duration)
                audio_data = np.interp(
                    np.linspace(0, duration, new_samples),
                    np.linspace(0, duration, len(audio_data)),
                    audio_data
                )
                duration = new_duration
            
            # Нормализуем
            if np.max(np.abs(audio_data)) > 0:
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
            
            # Создаем аудио сегмент
            segment = AudioSegment(
                start_time=0.0,
                end_time=duration,
                duration=duration,
                audio_data=audio_data.tobytes(),
                confidence=0.7,  # Средняя уверенность для fallback
                is_speech=True
            )
            
            self.logger.debug(f"Fallback синтезировал {segment.duration:.2f}s аудио")
            return segment
            
        except Exception as e:
            self.logger.error(f"Ошибка fallback синтеза: {e}")
            return None
    
    async def synthesize_batch(
        self, 
        texts: List[str],
        voice: Optional[str] = None,
        speed: Optional[float] = None
    ) -> List[AudioSegment]:
        """
        Пакетный синтез нескольких текстов
        
        Args:
            texts: Список текстов для синтеза
            voice: Голос
            speed: Скорость
            
        Returns:
            Список аудио сегментов
        """
        segments = []
        
        for text in texts:
            segment = await self.synthesize_speech(text, voice, speed)
            if segment:
                segments.append(segment)
        
        return segments
    
    async def synthesize_interview_response(
        self, 
        response_text: str,
        emotion: Optional[str] = None
    ) -> Optional[AudioSegment]:
        """
        Синтез ответа для интервью с учетом эмоций
        
        Args:
            response_text: Текст ответа
            emotion: Эмоциональная окраска (happy, sad, neutral, etc.)
            
        Returns:
            Аудио сегмент с ответом
        """
        try:
            # Настраиваем параметры в зависимости от эмоции
            voice_modifier = self._get_voice_modifier_for_emotion(emotion)
            
            # Синтезируем речь с модификаторами
            segment = await self.synthesize_speech(
                text=response_text,
                voice=self.voice,
                speed=self.speed * voice_modifier.get('speed', 1.0)
            )
            
            if segment and emotion:
                # Применяем эмоциональные модификации к аудио
                segment = await self._apply_emotion_modifications(segment, emotion)
            
            return segment
            
        except Exception as e:
            self.logger.error(f"Ошибка синтеза ответа интервью: {e}")
            return None
    
    def _get_voice_modifier_for_emotion(self, emotion: str) -> Dict[str, float]:
        """Получение модификаторов голоса для эмоции"""
        modifiers = {
            'happy': {'speed': 1.1, 'pitch': 1.2},
            'sad': {'speed': 0.9, 'pitch': 0.8},
            'angry': {'speed': 1.2, 'pitch': 1.3},
            'fearful': {'speed': 0.8, 'pitch': 1.1},
            'surprised': {'speed': 1.3, 'pitch': 1.4},
            'neutral': {'speed': 1.0, 'pitch': 1.0}
        }
        
        return modifiers.get(emotion, modifiers['neutral'])
    
    async def _apply_emotion_modifications(
        self, 
        segment: AudioSegment, 
        emotion: str
    ) -> AudioSegment:
        """
        Применение эмоциональных модификаций к аудио
        
        Args:
            segment: Исходный аудио сегмент
            emotion: Эмоция
            
        Returns:
            Модифицированный аудио сегмент
        """
        try:
            # Конвертируем аудио в numpy array
            audio_array = np.frombuffer(segment.audio_data, dtype=np.float32)
            
            # Применяем модификации в зависимости от эмоции
            if emotion == 'happy':
                # Увеличиваем яркость (спектральные характеристики)
                audio_array = self._enhance_spectral_features(audio_array, factor=1.2)
            elif emotion == 'sad':
                # Уменьшаем яркость
                audio_array = self._enhance_spectral_features(audio_array, factor=0.8)
            elif emotion == 'angry':
                # Добавляем резкость
                audio_array = self._add_sharpness(audio_array)
            elif emotion == 'fearful':
                # Добавляем дрожание
                audio_array = self._add_tremor(audio_array)
            
            # Создаем новый сегмент
            modified_segment = AudioSegment(
                start_time=segment.start_time,
                end_time=segment.end_time,
                duration=segment.duration,
                audio_data=audio_array.tobytes(),
                confidence=segment.confidence,
                is_speech=segment.is_speech
            )
            
            return modified_segment
            
        except Exception as e:
            self.logger.error(f"Ошибка применения эмоциональных модификаций: {e}")
            return segment
    
    def _enhance_spectral_features(self, audio: np.ndarray, factor: float) -> np.ndarray:
        """Усиление спектральных характеристик"""
        try:
            # Простое усиление высоких частот
            enhanced = audio.copy()
            
            # Применяем фильтр высоких частот
            for i in range(1, len(audio)):
                enhanced[i] = factor * audio[i] - 0.1 * audio[i-1]
            
            return enhanced
        except Exception:
            return audio
    
    def _add_sharpness(self, audio: np.ndarray) -> np.ndarray:
        """Добавление резкости к аудио"""
        try:
            # Простое добавление резкости через дифференцирование
            sharp = audio.copy()
            
            for i in range(1, len(audio)):
                sharp[i] = audio[i] + 0.3 * (audio[i] - audio[i-1])
            
            return sharp
        except Exception:
            return audio
    
    def _add_tremor(self, audio: np.ndarray) -> np.ndarray:
        """Добавление дрожания к аудио"""
        try:
            # Добавляем модуляцию амплитуды
            t = np.linspace(0, len(audio) / self.sample_rate, len(audio))
            tremor = 1.0 + 0.1 * np.sin(2 * np.pi * 8 * t)  # 8 Hz модуляция
            
            return audio * tremor
        except Exception:
            return audio
    
    def get_available_voices(self) -> List[str]:
        """Получение списка доступных голосов"""
        if self.piper_available:
            # В реальной реализации здесь нужно получить список голосов от Piper
            return [
                "ru_ru-oksana-medium",
                "ru_ru-oksana-high",
                "en_us-amy-medium",
                "en_us-amy-high"
            ]
        else:
            return ["fallback-voice"]
    
    def get_voice_info(self, voice: str) -> Dict[str, Any]:
        """Получение информации о голосе"""
        return {
            "name": voice,
            "language": voice.split('-')[0] if '-' in voice else "unknown",
            "quality": voice.split('-')[-1] if '-' in voice else "medium",
            "sample_rate": self.sample_rate,
            "available": self.piper_available
        }
    
    async def health_check(self) -> bool:
        """Проверка работоспособности сервиса"""
        try:
            # Тестируем синтез простого текста
            test_text = "Тест"
            result = await self.synthesize_speech(test_text)
            
            return result is not None
            
        except Exception as e:
            self.logger.error(f"Ошибка health check: {e}")
            return False
