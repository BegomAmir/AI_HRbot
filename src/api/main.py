"""
Главный FastAPI сервис - объединяет Gateway и Agent Runtime
"""
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from src.api.gateway import app as gateway_app, SessionManager
from src.runtime.agent_runtime import AgentRuntimeManager
from src.utils.logger import get_logger, setup_logger
from src.config.settings import settings

logger = get_logger(__name__)

# Глобальный менеджер runtime
runtime_manager: AgentRuntimeManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения"""
    global runtime_manager
    
    # Инициализация
    logger.info("Запуск AI HR Bot Service...")
    setup_logger(level="INFO")
    
    # Создаем менеджер runtime
    runtime_manager = AgentRuntimeManager()
    
    # Добавляем колбэк телеметрии
    runtime_manager.add_telemetry_callback(handle_telemetry)
    
    logger.info("AI HR Bot Service запущен")
    
    yield
    
    # Очистка
    logger.info("Остановка AI HR Bot Service...")
    
    # Останавливаем все runtime
    if runtime_manager:
        for session_id in list(runtime_manager.get_all_runtimes().keys()):
            await runtime_manager.stop_runtime(session_id)
    
    logger.info("AI HR Bot Service остановлен")


# Создаем главное приложение
app = FastAPI(
    title="AI HR Bot Service",
    description="Полнофункциональный сервис для проведения AI интервью",
    version="1.0.0",
    lifespan=lifespan
)

# Добавляем CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Включаем Gateway роуты
app.include_router(gateway_app.router, prefix="/gateway")

# Зависимости
def get_runtime_manager() -> AgentRuntimeManager:
    """Получение менеджера runtime"""
    return runtime_manager


def get_session_manager() -> SessionManager:
    """Получение менеджера сессий"""
    return SessionManager()


# Интеграция Gateway с Agent Runtime
async def handle_telemetry(telemetry_data: dict):
    """Обработка телеметрии от Agent Runtime"""
    try:
        session_id = telemetry_data.get("session_id")
        event_type = telemetry_data.get("event_type")
        
        logger.debug(f"Получена телеметрия: {event_type} для сессии {session_id}")
        
        # Здесь можно добавить логику обработки телеметрии
        # Например, сохранение в базу данных, отправка в внешние системы
        
    except Exception as e:
        logger.error(f"Ошибка обработки телеметрии: {e}")


# Переопределяем методы SessionManager для интеграции с Agent Runtime
class IntegratedSessionManager(SessionManager):
    """Интегрированный менеджер сессий с Agent Runtime"""
    
    def __init__(self):
        super().__init__()
        self.runtime_manager = None
    
    def set_runtime_manager(self, runtime_manager: AgentRuntimeManager):
        """Установка менеджера runtime"""
        self.runtime_manager = runtime_manager
    
    async def start_session(self, session_id: str) -> bool:
        """Запуск сессии с созданием Agent Runtime"""
        try:
            if session_id not in active_sessions:
                raise HTTPException(status_code=404, detail="Сессия не найдена")
            
            session_data = active_sessions[session_id]
            
            if session_data["status"] != "created":
                raise HTTPException(status_code=400, detail="Сессия уже запущена или завершена")
            
            # Создаем Agent Runtime
            if self.runtime_manager:
                runtime = await self.runtime_manager.create_runtime(session_id, session_data)
                success = await self.runtime_manager.start_runtime(session_id)
                
                if success:
                    session_data["status"] = "running"
                    session_data["updated_at"] = datetime.now()
                    self.logger.info(f"Сессия {session_id} запущена с Agent Runtime")
                    return True
                else:
                    self.logger.error(f"Не удалось запустить Agent Runtime для сессии {session_id}")
                    return False
            else:
                self.logger.error("Runtime Manager не инициализирован")
                return False
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Ошибка запуска сессии {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка запуска сессии: {str(e)}")
    
    async def stop_session(self, session_id: str) -> StopSessionResponse:
        """Остановка сессии с остановкой Agent Runtime"""
        try:
            if session_id not in active_sessions:
                raise HTTPException(status_code=404, detail="Сессия не найдена")
            
            session_data = active_sessions[session_id]
            
            if session_data["status"] == "stopped":
                raise HTTPException(status_code=400, detail="Сессия уже остановлена")
            
            # Останавливаем Agent Runtime
            if self.runtime_manager:
                await self.runtime_manager.stop_runtime(session_id)
            
            # Обновляем статус
            session_data["status"] = "stopped"
            session_data["updated_at"] = datetime.now()
            
            # Вычисляем финальные метрики
            duration = (session_data["updated_at"] - session_data["created_at"]).total_seconds()
            session_data["metrics"]["total_duration"] = duration
            
            self.logger.info(f"Сессия {session_id} остановлена, длительность: {duration:.2f}s")
            
            return StopSessionResponse(
                session_id=session_id,
                status="stopped",
                stopped_at=session_data["updated_at"],
                final_metrics=session_data["metrics"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Ошибка остановки сессии {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка остановки сессии: {str(e)}")


# Дополнительные API эндпоинты для управления Agent Runtime

@app.get("/api/runtime/status")
async def get_runtime_status(runtime_manager: AgentRuntimeManager = Depends(get_runtime_manager)):
    """Получение статуса всех Agent Runtime"""
    return runtime_manager.get_statistics()

@app.get("/api/runtime/{session_id}/status")
async def get_session_runtime_status(
    session_id: str,
    runtime_manager: AgentRuntimeManager = Depends(get_runtime_manager)
):
    """Получение статуса конкретного Agent Runtime"""
    runtime = runtime_manager.get_runtime(session_id)
    if not runtime:
        raise HTTPException(status_code=404, detail="Runtime не найден")
    
    return runtime.get_status()

@app.post("/api/runtime/{session_id}/response")
async def generate_bot_response(
    session_id: str,
    response_text: str,
    emotion: str = "neutral",
    runtime_manager: AgentRuntimeManager = Depends(get_runtime_manager)
):
    """Генерация ответа бота для конкретной сессии"""
    runtime = runtime_manager.get_runtime(session_id)
    if not runtime:
        raise HTTPException(status_code=404, detail="Runtime не найден")
    
    if not runtime.is_running:
        raise HTTPException(status_code=400, detail="Runtime не запущен")
    
    audio_response = await runtime.generate_bot_response(response_text, emotion)
    
    if audio_response:
        return {
            "success": True,
            "audio_duration": audio_response.duration,
            "message": "Ответ бота сгенерирован"
        }
    else:
        raise HTTPException(status_code=500, detail="Ошибка генерации ответа бота")

@app.get("/api/health")
async def health_check(runtime_manager: AgentRuntimeManager = Depends(get_runtime_manager)):
    """Расширенная проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "runtime_manager": runtime_manager.get_statistics() if runtime_manager else None,
        "services": {
            "gateway": "active",
            "agent_runtime": "active" if runtime_manager else "inactive",
            "livekit": "placeholder",  # TODO: проверка LiveKit
            "llm": "placeholder"  # TODO: проверка LLM
        }
    }

# Инициализация интегрированного менеджера сессий
@app.on_event("startup")
async def startup_event():
    """Событие запуска приложения"""
    global runtime_manager
    
    # Создаем интегрированный менеджер сессий
    integrated_manager = IntegratedSessionManager()
    integrated_manager.set_runtime_manager(runtime_manager)
    
    # Заменяем менеджер в Gateway
    gateway_app.dependency_overrides[get_session_manager] = lambda: integrated_manager


# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
