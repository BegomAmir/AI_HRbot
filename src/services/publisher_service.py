"""
Сервис Publisher - публикация аудио бота
Отправляет сгенерированные аудио ответы бота в LiveKit или другие системы
"""
import asyncio
import json
from typing import List, Optional, Dict, Any, Callable
from dataclasses import dataclass
from src.models.interview import AudioSegment
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


@dataclass
class PublisherConfig:
    """Конфигурация для publisher"""
    enable_livekit: bool = True
    enable_file_output: bool = False
    output_directory: str = "output"
    enable_webhook: bool = False
    webhook_url: str = ""
    enable_metrics: bool = True


@dataclass
class PublishResult:
    """Результат публикации"""
    success: bool
    message: str
    timestamp: float
    audio_duration: float
    destination: str


class PublisherService:
    """
    Сервис для публикации аудио бота
    
    Поддерживает:
    - LiveKit для реального времени
    - Файловый вывод для отладки
    - Webhook для интеграции
    - Метрики для мониторинга
    """
    
    def __init__(self, config: Optional[PublisherConfig] = None):
        self.logger = get_logger(f"{__name__}.PublisherService")
        self.config = config or PublisherConfig()
        
        # Колбэки для различных типов публикации
        self.livekit_callbacks = []
        self.file_callbacks = []
        self.webhook_callbacks = []
        self.metrics_callbacks = []
        
        # Статистика
        self.published_count = 0
        self.failed_count = 0
        self.total_duration = 0.0
        
        self.logger.info("Publisher сервис инициализирован")
    
    async def publish_audio(
        self, 
        audio_segment: AudioSegment,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PublishResult:
        """
        Публикация аудио сегмента
        
        Args:
            audio_segment: Аудио сегмент для публикации
            metadata: Дополнительные метаданные
            
        Returns:
            Результат публикации
        """
        try:
            if not audio_segment or not audio_segment.audio_data:
                return PublishResult(
                    success=False,
                    message="Пустой аудио сегмент",
                    timestamp=asyncio.get_event_loop().time(),
                    audio_duration=0.0,
                    destination="none"
                )
            
            self.logger.info(f"Публикация аудио: {audio_segment.duration:.2f}s")
            
            # Подготавливаем метаданные
            publish_metadata = metadata or {}
            publish_metadata.update({
                "duration": audio_segment.duration,
                "start_time": audio_segment.start_time,
                "end_time": audio_segment.end_time,
                "confidence": audio_segment.confidence,
                "is_speech": audio_segment.is_speech
            })
            
            # Публикуем в различные системы
            results = []
            
            if self.config.enable_livekit:
                result = await self._publish_to_livekit(audio_segment, publish_metadata)
                results.append(result)
            
            if self.config.enable_file_output:
                result = await self._publish_to_file(audio_segment, publish_metadata)
                results.append(result)
            
            if self.config.enable_webhook:
                result = await self._publish_to_webhook(audio_segment, publish_metadata)
                results.append(result)
            
            if self.config.enable_metrics:
                result = await self._publish_metrics(audio_segment, publish_metadata)
                results.append(result)
            
            # Определяем общий результат
            success_count = sum(1 for r in results if r.success)
            total_count = len(results)
            
            if success_count == total_count:
                self.published_count += 1
                self.total_duration += audio_segment.duration
                
                return PublishResult(
                    success=True,
                    message=f"Успешно опубликовано в {success_count}/{total_count} систем",
                    timestamp=asyncio.get_event_loop().time(),
                    audio_duration=audio_segment.duration,
                    destination="multiple"
                )
            else:
                self.failed_count += 1
                
                return PublishResult(
                    success=False,
                    message=f"Ошибка публикации: {success_count}/{total_count} успешно",
                    timestamp=asyncio.get_event_loop().time(),
                    audio_duration=audio_segment.duration,
                    destination="multiple"
                )
                
        except Exception as e:
            self.logger.error(f"Ошибка публикации аудио: {e}")
            self.failed_count += 1
            
            return PublishResult(
                success=False,
                message=f"Критическая ошибка: {str(e)}",
                timestamp=asyncio.get_event_loop().time(),
                audio_duration=audio_segment.duration if audio_segment else 0.0,
                destination="error"
            )
    
    async def _publish_to_livekit(
        self, 
        audio_segment: AudioSegment,
        metadata: Dict[str, Any]
    ) -> PublishResult:
        """Публикация в LiveKit"""
        try:
            # Здесь должна быть интеграция с LiveKit
            # Пока что симулируем успешную публикацию
            
            self.logger.debug("Публикация в LiveKit...")
            
            # Уведомляем колбэки
            for callback in self.livekit_callbacks:
                try:
                    await callback(audio_segment, metadata)
                except Exception as e:
                    self.logger.error(f"Ошибка в LiveKit колбэке: {e}")
            
            # Симуляция задержки
            await asyncio.sleep(0.01)
            
            return PublishResult(
                success=True,
                message="Успешно опубликовано в LiveKit",
                timestamp=asyncio.get_event_loop().time(),
                audio_duration=audio_segment.duration,
                destination="livekit"
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка публикации в LiveKit: {e}")
            return PublishResult(
                success=False,
                message=f"Ошибка LiveKit: {str(e)}",
                timestamp=asyncio.get_event_loop().time(),
                audio_duration=audio_segment.duration,
                destination="livekit"
            )
    
    async def _publish_to_file(
        self, 
        audio_segment: AudioSegment,
        metadata: Dict[str, Any]
    ) -> PublishResult:
        """Публикация в файл"""
        try:
            import os
            import soundfile as sf
            import numpy as np
            
            # Создаем директорию если не существует
            os.makedirs(self.config.output_directory, exist_ok=True)
            
            # Генерируем имя файла
            timestamp = int(asyncio.get_event_loop().time() * 1000)
            filename = f"bot_audio_{timestamp}.wav"
            filepath = os.path.join(self.config.output_directory, filename)
            
            # Конвертируем аудио данные
            audio_array = np.frombuffer(audio_segment.audio_data, dtype=np.float32)
            
            # Сохраняем аудио файл
            sf.write(filepath, audio_array, 16000)
            
            # Сохраняем метаданные
            metadata_file = filepath.replace('.wav', '.json')
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            
            self.logger.debug(f"Аудио сохранено в файл: {filepath}")
            
            # Уведомляем колбэки
            for callback in self.file_callbacks:
                try:
                    await callback(audio_segment, metadata, filepath)
                except Exception as e:
                    self.logger.error(f"Ошибка в файловом колбэке: {e}")
            
            return PublishResult(
                success=True,
                message=f"Успешно сохранено в файл: {filename}",
                timestamp=asyncio.get_event_loop().time(),
                audio_duration=audio_segment.duration,
                destination="file"
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения в файл: {e}")
            return PublishResult(
                success=False,
                message=f"Ошибка файла: {str(e)}",
                timestamp=asyncio.get_event_loop().time(),
                audio_duration=audio_segment.duration,
                destination="file"
            )
    
    async def _publish_to_webhook(
        self, 
        audio_segment: AudioSegment,
        metadata: Dict[str, Any]
    ) -> PublishResult:
        """Публикация через webhook"""
        try:
            import aiohttp
            
            if not self.config.webhook_url:
                return PublishResult(
                    success=False,
                    message="Webhook URL не настроен",
                    timestamp=asyncio.get_event_loop().time(),
                    audio_duration=audio_segment.duration,
                    destination="webhook"
                )
            
            # Подготавливаем данные для webhook
            webhook_data = {
                "audio_duration": audio_segment.duration,
                "metadata": metadata,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            # Отправляем webhook
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.webhook_url,
                    json=webhook_data,
                    timeout=aiohttp.ClientTimeout(total=5.0)
                ) as response:
                    if response.status == 200:
                        self.logger.debug("Webhook отправлен успешно")
                        
                        # Уведомляем колбэки
                        for callback in self.webhook_callbacks:
                            try:
                                await callback(audio_segment, metadata, response.status)
                            except Exception as e:
                                self.logger.error(f"Ошибка в webhook колбэке: {e}")
                        
                        return PublishResult(
                            success=True,
                            message="Webhook отправлен успешно",
                            timestamp=asyncio.get_event_loop().time(),
                            audio_duration=audio_segment.duration,
                            destination="webhook"
                        )
                    else:
                        return PublishResult(
                            success=False,
                            message=f"Webhook ошибка: HTTP {response.status}",
                            timestamp=asyncio.get_event_loop().time(),
                            audio_duration=audio_segment.duration,
                            destination="webhook"
                        )
            
        except Exception as e:
            self.logger.error(f"Ошибка webhook: {e}")
            return PublishResult(
                success=False,
                message=f"Ошибка webhook: {str(e)}",
                timestamp=asyncio.get_event_loop().time(),
                audio_duration=audio_segment.duration,
                destination="webhook"
            )
    
    async def _publish_metrics(
        self, 
        audio_segment: AudioSegment,
        metadata: Dict[str, Any]
    ) -> PublishResult:
        """Публикация метрик"""
        try:
            # Собираем метрики
            metrics = {
                "audio_duration": audio_segment.duration,
                "confidence": audio_segment.confidence,
                "timestamp": asyncio.get_event_loop().time(),
                "published_count": self.published_count,
                "failed_count": self.failed_count,
                "total_duration": self.total_duration
            }
            
            # Уведомляем колбэки метрик
            for callback in self.metrics_callbacks:
                try:
                    await callback(metrics)
                except Exception as e:
                    self.logger.error(f"Ошибка в метриках колбэке: {e}")
            
            self.logger.debug("Метрики обновлены")
            
            return PublishResult(
                success=True,
                message="Метрики обновлены",
                timestamp=asyncio.get_event_loop().time(),
                audio_duration=audio_segment.duration,
                destination="metrics"
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка публикации метрик: {e}")
            return PublishResult(
                success=False,
                message=f"Ошибка метрик: {str(e)}",
                timestamp=asyncio.get_event_loop().time(),
                audio_duration=audio_segment.duration,
                destination="metrics"
            )
    
    def add_livekit_callback(self, callback: Callable):
        """Добавление колбэка для LiveKit"""
        self.livekit_callbacks.append(callback)
    
    def add_file_callback(self, callback: Callable):
        """Добавление колбэка для файлов"""
        self.file_callbacks.append(callback)
    
    def add_webhook_callback(self, callback: Callable):
        """Добавление колбэка для webhook"""
        self.webhook_callbacks.append(callback)
    
    def add_metrics_callback(self, callback: Callable):
        """Добавление колбэка для метрик"""
        self.metrics_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики публикации"""
        return {
            "published_count": self.published_count,
            "failed_count": self.failed_count,
            "total_duration": self.total_duration,
            "success_rate": self.published_count / (self.published_count + self.failed_count) if (self.published_count + self.failed_count) > 0 else 0.0
        }
    
    def update_config(self, config: PublisherConfig):
        """Обновление конфигурации"""
        self.config = config
        self.logger.info("Конфигурация publisher обновлена")
    
    def get_config(self) -> PublisherConfig:
        """Получение текущей конфигурации"""
        return self.config
    
    async def health_check(self) -> bool:
        """Проверка работоспособности сервиса"""
        try:
            # Создаем тестовый аудио сегмент
            test_segment = AudioSegment(
                start_time=0.0,
                end_time=0.1,
                duration=0.1,
                audio_data=b'\x00' * 100,
                confidence=0.8,
                is_speech=True
            )
            
            # Пытаемся опубликовать (только метрики для теста)
            original_config = self.config
            self.config = PublisherConfig(
                enable_livekit=False,
                enable_file_output=False,
                enable_webhook=False,
                enable_metrics=True
            )
            
            result = await self.publish_audio(test_segment)
            
            # Восстанавливаем конфигурацию
            self.config = original_config
            
            return result.success
            
        except Exception as e:
            self.logger.error(f"Ошибка health check: {e}")
            return False
