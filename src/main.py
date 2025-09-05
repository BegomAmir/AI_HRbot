"""
Главный файл Python Agent для AI HR Bot
Объединяет все микросервисы в единую систему
"""
import asyncio
import signal
import sys
from typing import List, Optional
from src.services.audio_processing import AudioProcessor, AudioBuffer
from src.services.vad_service import SileroVADService
from src.services.hybrid_stt_service import HybridSTTService
from src.services.prosody_service import ProsodyAnalysisService
from src.services.emotion_service import EmotionRecognitionService
from src.services.tts_service import TTSService
from src.models.interview import AudioSegment, SpeechTranscription, ProsodyFeatures, EmotionAnalysis
from src.utils.logger import get_logger, setup_logger
from src.config.settings import settings

logger = get_logger(__name__)


class AIHRAgent:
    """
    Основной класс AI HR Agent
    
    Координирует работу всех сервисов:
    - Обработка аудио
    - VAD (Voice Activity Detection)
    - STT (Speech-to-Text)
    - Анализ просодии
    - Распознавание эмоций
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.AIHRAgent")
        
        # Инициализация сервисов
        self.audio_processor = AudioProcessor()
        self.vad_service = SileroVADService()
        self.stt_service = HybridSTTService()  # Используем гибридный STT
        self.prosody_service = ProsodyAnalysisService()
        self.emotion_service = EmotionRecognitionService()
        self.tts_service = TTSService()
        
        # Аудио буфер
        self.audio_buffer = AudioBuffer(max_duration=10.0)
        
        # Состояние агента
        self.is_running = False
        self.current_session = None
        
        # Обработчики событий
        self.audio_callbacks = []
        self.transcription_callbacks = []
        self.analysis_callbacks = []
        
        self.logger.info("AI HR Agent инициализирован")
    
    async def start(self):
        """Запуск агента"""
        try:
            self.logger.info("Запуск AI HR Agent...")
            
            # Проверяем работоспособность всех сервисов
            await self._health_check()
            
            self.is_running = True
            
            # Регистрируем обработчики сигналов
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            self.logger.info("AI HR Agent запущен и готов к работе")
            
            # Основной цикл
            while self.is_running:
                await asyncio.sleep(1)
                
        except Exception as e:
            self.logger.error(f"Ошибка запуска агента: {e}")
            await self.stop()
    
    async def stop(self):
        """Остановка агента"""
        self.logger.info("Остановка AI HR Agent...")
        self.is_running = False
        
        # Очищаем буферы
        self.audio_buffer.clear()
        
        self.logger.info("AI HR Agent остановлен")
    
    def _signal_handler(self, signum, frame):
        """Обработчик сигналов для корректного завершения"""
        self.logger.info(f"Получен сигнал {signum}, завершение работы...")
        asyncio.create_task(self.stop())
    
    async def _health_check(self):
        """Проверка работоспособности всех сервисов"""
        self.logger.info("Проверка работоспособности сервисов...")
        
        services = [
            ("VAD", self.vad_service.health_check()),
            ("STT", self.stt_service.health_check()),
            ("Prosody", self.prosody_service.health_check()),
            ("Emotion", self.emotion_service.health_check()),
            ("TTS", self.tts_service.health_check())
        ]
        
        for service_name, health_check in services:
            try:
                is_healthy = await health_check
                if is_healthy:
                    self.logger.info(f"Сервис {service_name}: OK")
                else:
                    self.logger.warning(f"Сервис {service_name}: WARNING")
            except Exception as e:
                self.logger.error(f"Сервис {service_name}: ERROR - {e}")
    
    async def process_audio_stream(
        self, 
        audio_data: bytes,
        timestamp: float = 0.0
    ):
        """
        Обработка входящего аудио потока
        
        Args:
            audio_data: Сырые аудио данные
            timestamp: Временная метка
        """
        try:
            if not self.is_running:
                return
            
            # Обрабатываем аудио чанк
            audio_segment = await self.audio_processor.process_audio_chunk(
                audio_data, timestamp
            )
            
            # Добавляем в буфер
            self.audio_buffer.add_segment(audio_segment)
            
            # Уведомляем колбэки
            for callback in self.audio_callbacks:
                try:
                    await callback(audio_segment)
                except Exception as e:
                    self.logger.error(f"Ошибка в аудио колбэке: {e}")
            
            # Если буфер достаточно заполнен, запускаем анализ
            if self.audio_buffer.get_buffer_duration() >= 2.0:  # 2 секунды
                await self._analyze_audio_buffer()
                
        except Exception as e:
            self.logger.error(f"Ошибка обработки аудио потока: {e}")
    
    async def _analyze_audio_buffer(self):
        """Анализ накопленного аудио буфера"""
        try:
            # Получаем все сегменты из буфера
            segments = self.audio_buffer.buffer.copy()
            
            if not segments:
                return
            
            self.logger.debug(f"Анализ {len(segments)} аудио сегментов")
            
            # 1. VAD - обнаружение речи
            speech_segments = await self.vad_service.process_audio_stream(segments)
            
            if not speech_segments:
                self.logger.debug("Речь не обнаружена")
                return
            
            # 2. STT - транскрипция речи
            transcriptions = await self.stt_service.transcribe_batch(speech_segments)
            
            # 3. Анализ просодии
            prosody_features = await self.prosody_service.analyze_batch(speech_segments)
            
            # 4. Распознавание эмоций
            emotion_analysis = await self.emotion_service.analyze_batch(speech_segments)
            
            # Уведомляем колбэки об анализе
            for callback in self.analysis_callbacks:
                try:
                    await callback({
                        'transcriptions': transcriptions,
                        'prosody': prosody_features,
                        'emotions': emotion_analysis,
                        'timestamp': asyncio.get_event_loop().time()
                    })
                except Exception as e:
                    self.logger.error(f"Ошибка в колбэке анализа: {e}")
            
            # Очищаем буфер после анализа
            self.audio_buffer.clear()
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа аудио буфера: {e}")
    
    def add_audio_callback(self, callback):
        """Добавление колбэка для аудио событий"""
        self.audio_callbacks.append(callback)
    
    def add_transcription_callback(self, callback):
        """Добавление колбэка для транскрипций"""
        self.transcription_callbacks.append(callback)
    
    def add_analysis_callback(self, callback):
        """Добавление колбэка для результатов анализа"""
        self.analysis_callbacks.append(callback)
    
    async def get_status(self) -> dict:
        """Получение статуса агента"""
        return {
            'is_running': self.is_running,
            'buffer_duration': self.audio_buffer.get_buffer_duration(),
            'services': {
                'vad': 'active',
                'stt': 'active',
                'prosody': 'active',
                'emotion': 'active',
                'tts': 'active'
            }
        }
    
    async def generate_bot_response(
        self, 
        response_text: str,
        emotion: Optional[str] = None
    ) -> Optional[AudioSegment]:
        """
        Генерация голосового ответа бота
        
        Args:
            response_text: Текст ответа
            emotion: Эмоциональная окраска
            
        Returns:
            Аудио сегмент с ответом бота или None при ошибке
        """
        try:
            if not self.is_running:
                return None
            
            self.logger.info(f"Генерация ответа бота: '{response_text[:50]}...'")
            
            # Синтезируем речь
            audio_response = await self.tts_service.synthesize_interview_response(
                response_text, emotion
            )
            
            if audio_response:
                self.logger.info(f"Ответ бота сгенерирован: {audio_response.duration:.2f}s")
                
                # Уведомляем колбэки о сгенерированном ответе
                for callback in self.audio_callbacks:
                    try:
                        await callback(audio_response)
                    except Exception as e:
                        self.logger.error(f"Ошибка в колбэке ответа бота: {e}")
                
                return audio_response
            else:
                self.logger.error("Не удалось сгенерировать ответ бота")
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка генерации ответа бота: {e}")
            return None
    
    async def transcribe_with_speakers(
        self, 
        audio_segment: AudioSegment,
        language: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Транскрипция с разделением спикеров (если доступно)
        
        Args:
            audio_segment: Аудио сегмент
            language: Язык речи
            
        Returns:
            Результат с разделением спикеров или None
        """
        try:
            if not self.is_running:
                return None
            
            self.logger.info("Транскрипция с разделением спикеров...")
            
            result = await self.stt_service.transcribe_with_speakers(audio_segment, language)
            
            if result:
                self.logger.info(f"Транскрипция с спикерами выполнена: {len(result.get('speaker_segments', []))} сегментов")
                return result
            else:
                self.logger.warning("Диаризация недоступна или не удалась")
                return None
                
        except Exception as e:
            self.logger.error(f"Ошибка транскрипции с спикерами: {e}")
            return None
    
    def get_stt_capabilities(self) -> Dict[str, Any]:
        """Получение возможностей STT сервиса"""
        try:
            if hasattr(self.stt_service, 'get_capabilities'):
                return self.stt_service.get_capabilities()
            return {"basic_transcription": True}
        except Exception as e:
            self.logger.error(f"Ошибка получения возможностей STT: {e}")
            return {"basic_transcription": True}


async def main():
    """Главная функция"""
    # Настройка логирования
    setup_logger()
    
    # Создание и запуск агента
    agent = AIHRAgent()
    
    try:
        await agent.start()
    except KeyboardInterrupt:
        logger.info("Получен сигнал прерывания")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")
    finally:
        await agent.stop()


if __name__ == "__main__":
    # Запуск в асинхронном режиме
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Программа завершена пользователем")
    except Exception as e:
        logger.error(f"Ошибка запуска: {e}")
        sys.exit(1)
