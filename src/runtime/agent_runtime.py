"""
Agent Runtime - управление ботом-участником в LiveKit комнате
"""
import asyncio
import json
from typing import Dict, Any, Optional, Callable
from datetime import datetime
from src.main import AIHRAgent
from src.models.interview import AudioSegment, SpeechTranscription
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class AgentRuntime:
    """
    Runtime для управления агентом-ботом в LiveKit комнате
    
    Функции:
    - Подключение к LiveKit комнате
    - Управление жизненным циклом агента
    - Обработка аудио потока
    - Отправка телеметрии в Gateway
    """
    
    def __init__(self, session_id: str, session_data: Dict[str, Any]):
        self.session_id = session_id
        self.session_data = session_data
        self.logger = get_logger(f"{__name__}.AgentRuntime.{session_id}")
        
        # Агент
        self.agent: Optional[AIHRAgent] = None
        self.is_running = False
        
        # LiveKit соединение (заглушка)
        self.livekit_connection = None
        self.room_name = session_data["room_name"]
        
        # Телеметрия
        self.telemetry_callbacks: list[Callable] = []
        self.metrics = {
            "audio_chunks_processed": 0,
            "speech_segments_detected": 0,
            "transcriptions_completed": 0,
            "emotions_detected": 0,
            "bot_responses_generated": 0,
            "total_processing_time": 0.0,
            "last_activity": None
        }
        
        self.logger.info(f"Agent Runtime создан для сессии {session_id}")
    
    async def start(self) -> bool:
        """Запуск агента и подключение к LiveKit"""
        try:
            self.logger.info(f"Запуск Agent Runtime для сессии {self.session_id}")
            
            # Создаем агента
            self.agent = AIHRAgent()
            
            # Добавляем колбэки для телеметрии
            self.agent.add_audio_callback(self._on_audio_processed)
            self.agent.add_analysis_callback(self._on_analysis_completed)
            
            # Запускаем агента
            await self.agent.start()
            
            # Подключаемся к LiveKit (заглушка)
            await self._connect_to_livekit()
            
            self.is_running = True
            self.metrics["last_activity"] = datetime.now()
            
            self.logger.info(f"Agent Runtime запущен для сессии {self.session_id}")
            
            # Отправляем телеметрию о запуске
            await self._send_telemetry("agent_started", {
                "session_id": self.session_id,
                "room_name": self.room_name,
                "timestamp": datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка запуска Agent Runtime: {e}")
            await self.stop()
            return False
    
    async def stop(self) -> bool:
        """Остановка агента и отключение от LiveKit"""
        try:
            self.logger.info(f"Остановка Agent Runtime для сессии {self.session_id}")
            
            self.is_running = False
            
            # Останавливаем агента
            if self.agent:
                await self.agent.stop()
                self.agent = None
            
            # Отключаемся от LiveKit
            await self._disconnect_from_livekit()
            
            # Отправляем финальную телеметрию
            await self._send_telemetry("agent_stopped", {
                "session_id": self.session_id,
                "final_metrics": self.metrics,
                "timestamp": datetime.now().isoformat()
            })
            
            self.logger.info(f"Agent Runtime остановлен для сессии {self.session_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Ошибка остановки Agent Runtime: {e}")
            return False
    
    async def _connect_to_livekit(self):
        """Подключение к LiveKit комнате (заглушка)"""
        try:
            # TODO: Реальная интеграция с LiveKit
            self.logger.info(f"Подключение к LiveKit комнате: {self.room_name}")
            
            # Симулируем подключение
            await asyncio.sleep(0.1)
            
            self.livekit_connection = {
                "room_name": self.room_name,
                "connected": True,
                "connected_at": datetime.now()
            }
            
            self.logger.info(f"Подключен к LiveKit комнате: {self.room_name}")
            
        except Exception as e:
            self.logger.error(f"Ошибка подключения к LiveKit: {e}")
            raise
    
    async def _disconnect_from_livekit(self):
        """Отключение от LiveKit комнаты"""
        try:
            if self.livekit_connection:
                self.logger.info(f"Отключение от LiveKit комнаты: {self.room_name}")
                self.livekit_connection = None
                
        except Exception as e:
            self.logger.error(f"Ошибка отключения от LiveKit: {e}")
    
    async def process_audio_chunk(self, audio_data: bytes, timestamp: float):
        """Обработка аудио чанка от кандидата"""
        try:
            if not self.is_running or not self.agent:
                return
            
            # Отправляем аудио в агента
            await self.agent.process_audio_stream(audio_data, timestamp)
            
            # Обновляем метрики
            self.metrics["audio_chunks_processed"] += 1
            self.metrics["last_activity"] = datetime.now()
            
            # Отправляем телеметрию
            await self._send_telemetry("audio_processed", {
                "session_id": self.session_id,
                "chunk_size": len(audio_data),
                "timestamp": timestamp,
                "total_chunks": self.metrics["audio_chunks_processed"]
            })
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки аудио чанка: {e}")
    
    async def generate_bot_response(self, response_text: str, emotion: Optional[str] = None):
        """Генерация ответа бота"""
        try:
            if not self.is_running or not self.agent:
                return None
            
            # Генерируем ответ
            audio_response = await self.agent.generate_bot_response(response_text, emotion)
            
            if audio_response:
                # Обновляем метрики
                self.metrics["bot_responses_generated"] += 1
                self.metrics["last_activity"] = datetime.now()
                
                # Отправляем телеметрию
                await self._send_telemetry("bot_response_generated", {
                    "session_id": self.session_id,
                    "response_text": response_text,
                    "emotion": emotion,
                    "audio_duration": audio_response.duration,
                    "total_responses": self.metrics["bot_responses_generated"]
                })
                
                # TODO: Отправить аудио в LiveKit комнату
                await self._publish_audio_to_livekit(audio_response)
            
            return audio_response
            
        except Exception as e:
            self.logger.error(f"Ошибка генерации ответа бота: {e}")
            return None
    
    async def _publish_audio_to_livekit(self, audio_segment: AudioSegment):
        """Публикация аудио в LiveKit комнату (заглушка)"""
        try:
            # TODO: Реальная публикация в LiveKit
            self.logger.debug(f"Публикация аудио в LiveKit: {audio_segment.duration:.2f}s")
            
            # Симулируем публикацию
            await asyncio.sleep(0.01)
            
        except Exception as e:
            self.logger.error(f"Ошибка публикации аудио в LiveKit: {e}")
    
    async def _on_audio_processed(self, audio_segment: AudioSegment):
        """Колбэк обработки аудио"""
        try:
            # Отправляем телеметрию о обработанном аудио
            await self._send_telemetry("audio_segment_processed", {
                "session_id": self.session_id,
                "duration": audio_segment.duration,
                "confidence": audio_segment.confidence,
                "is_speech": audio_segment.is_speech,
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Ошибка в колбэке аудио: {e}")
    
    async def _on_analysis_completed(self, analysis_data: Dict[str, Any]):
        """Колбэк завершения анализа"""
        try:
            # Обновляем метрики
            if analysis_data.get("transcriptions"):
                self.metrics["transcriptions_completed"] += len(analysis_data["transcriptions"])
            
            if analysis_data.get("emotions"):
                self.metrics["emotions_detected"] += len(analysis_data["emotions"])
            
            if analysis_data.get("endpoints"):
                self.metrics["speech_segments_detected"] += len(analysis_data["endpoints"])
            
            self.metrics["last_activity"] = datetime.now()
            
            # Отправляем телеметрию
            await self._send_telemetry("analysis_completed", {
                "session_id": self.session_id,
                "transcriptions_count": len(analysis_data.get("transcriptions", [])),
                "emotions_count": len(analysis_data.get("emotions", [])),
                "endpoints_count": len(analysis_data.get("endpoints", [])),
                "timestamp": datetime.now().isoformat()
            })
            
        except Exception as e:
            self.logger.error(f"Ошибка в колбэке анализа: {e}")
    
    async def _send_telemetry(self, event_type: str, data: Dict[str, Any]):
        """Отправка телеметрии в Gateway"""
        try:
            telemetry_data = {
                "event_type": event_type,
                "session_id": self.session_id,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            # Уведомляем все колбэки
            for callback in self.telemetry_callbacks:
                try:
                    await callback(telemetry_data)
                except Exception as e:
                    self.logger.error(f"Ошибка в телеметрии колбэке: {e}")
            
            self.logger.debug(f"Телеметрия отправлена: {event_type}")
            
        except Exception as e:
            self.logger.error(f"Ошибка отправки телеметрии: {e}")
    
    def add_telemetry_callback(self, callback: Callable):
        """Добавление колбэка телеметрии"""
        self.telemetry_callbacks.append(callback)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Получение текущих метрик"""
        return self.metrics.copy()
    
    def get_status(self) -> Dict[str, Any]:
        """Получение статуса runtime"""
        return {
            "session_id": self.session_id,
            "is_running": self.is_running,
            "room_name": self.room_name,
            "livekit_connected": self.livekit_connection is not None,
            "agent_running": self.agent is not None and self.agent.is_running,
            "metrics": self.metrics,
            "last_activity": self.metrics["last_activity"]
        }


class AgentRuntimeManager:
    """Менеджер для управления множественными Agent Runtime"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.AgentRuntimeManager")
        self.runtimes: Dict[str, AgentRuntime] = {}
        self.telemetry_callbacks: list[Callable] = []
    
    async def create_runtime(self, session_id: str, session_data: Dict[str, Any]) -> AgentRuntime:
        """Создание нового runtime"""
        try:
            if session_id in self.runtimes:
                raise ValueError(f"Runtime для сессии {session_id} уже существует")
            
            runtime = AgentRuntime(session_id, session_data)
            
            # Добавляем колбэк телеметрии
            runtime.add_telemetry_callback(self._handle_telemetry)
            
            self.runtimes[session_id] = runtime
            self.logger.info(f"Создан runtime для сессии {session_id}")
            
            return runtime
            
        except Exception as e:
            self.logger.error(f"Ошибка создания runtime: {e}")
            raise
    
    async def start_runtime(self, session_id: str) -> bool:
        """Запуск runtime"""
        try:
            if session_id not in self.runtimes:
                raise ValueError(f"Runtime для сессии {session_id} не найден")
            
            runtime = self.runtimes[session_id]
            success = await runtime.start()
            
            if success:
                self.logger.info(f"Runtime для сессии {session_id} запущен")
            else:
                self.logger.error(f"Не удалось запустить runtime для сессии {session_id}")
                # Удаляем неудачный runtime
                del self.runtimes[session_id]
            
            return success
            
        except Exception as e:
            self.logger.error(f"Ошибка запуска runtime: {e}")
            return False
    
    async def stop_runtime(self, session_id: str) -> bool:
        """Остановка runtime"""
        try:
            if session_id not in self.runtimes:
                self.logger.warning(f"Runtime для сессии {session_id} не найден")
                return True
            
            runtime = self.runtimes[session_id]
            success = await runtime.stop()
            
            # Удаляем runtime
            del self.runtimes[session_id]
            
            if success:
                self.logger.info(f"Runtime для сессии {session_id} остановлен")
            else:
                self.logger.error(f"Ошибка остановки runtime для сессии {session_id}")
            
            return success
            
        except Exception as e:
            self.logger.error(f"Ошибка остановки runtime: {e}")
            return False
    
    def get_runtime(self, session_id: str) -> Optional[AgentRuntime]:
        """Получение runtime по ID сессии"""
        return self.runtimes.get(session_id)
    
    def get_all_runtimes(self) -> Dict[str, AgentRuntime]:
        """Получение всех runtime"""
        return self.runtimes.copy()
    
    async def _handle_telemetry(self, telemetry_data: Dict[str, Any]):
        """Обработка телеметрии от runtime"""
        try:
            # Пересылаем телеметрию в Gateway
            for callback in self.telemetry_callbacks:
                try:
                    await callback(telemetry_data)
                except Exception as e:
                    self.logger.error(f"Ошибка в телеметрии колбэке: {e}")
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки телеметрии: {e}")
    
    def add_telemetry_callback(self, callback: Callable):
        """Добавление колбэка телеметрии"""
        self.telemetry_callbacks.append(callback)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Получение статистики менеджера"""
        return {
            "total_runtimes": len(self.runtimes),
            "active_runtimes": sum(1 for r in self.runtimes.values() if r.is_running),
            "runtime_sessions": list(self.runtimes.keys())
        }
