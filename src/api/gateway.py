"""
FastAPI Gateway - REST API для управления интервью сессиями
"""
import asyncio
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from src.models.interview import InterviewSession, InterviewStatus
from src.utils.logger import get_logger, setup_logger
from src.config.settings import settings

logger = get_logger(__name__)

# Pydantic модели для API
class CreateSessionRequest(BaseModel):
    """Запрос на создание сессии"""
    vacancy_id: str = Field(..., description="ID вакансии")
    resume_id: str = Field(..., description="ID резюме")
    candidate_name: str = Field(..., description="Имя кандидата")
    interview_duration: int = Field(default=30, description="Длительность интервью в минутах")
    language: str = Field(default="ru", description="Язык интервью")

class SessionResponse(BaseModel):
    """Ответ с информацией о сессии"""
    session_id: str
    room_name: str
    livekit_url: str
    client_token: str
    agent_token: str
    status: str
    created_at: datetime
    expires_at: datetime

class SessionStatusResponse(BaseModel):
    """Статус сессии"""
    session_id: str
    status: str
    duration: float
    metrics: Dict[str, Any]
    created_at: datetime
    updated_at: datetime

class StopSessionResponse(BaseModel):
    """Ответ на остановку сессии"""
    session_id: str
    status: str
    stopped_at: datetime
    final_metrics: Dict[str, Any]

class VacancyRequest(BaseModel):
    """Запрос на создание/обновление вакансии"""
    title: str
    description: str
    requirements: List[str]
    skills: List[str]
    experience_level: str
    salary_range: Optional[str] = None

class ResumeRequest(BaseModel):
    """Запрос на создание/обновление резюме"""
    candidate_name: str
    email: str
    phone: Optional[str] = None
    experience: List[Dict[str, Any]]
    skills: List[str]
    education: List[Dict[str, Any]]

class ReportRequest(BaseModel):
    """Запрос на генерацию отчета"""
    session_id: str
    include_audio: bool = False
    include_transcript: bool = True
    include_metrics: bool = True

# Глобальное хранилище сессий (в продакшене использовать Redis/DB)
active_sessions: Dict[str, Dict[str, Any]] = {}
vacancies: Dict[str, Dict[str, Any]] = {}
resumes: Dict[str, Dict[str, Any]] = {}

# Создаем FastAPI приложение
app = FastAPI(
    title="AI HR Bot Gateway",
    description="Gateway для управления интервью сессиями с AI HR Bot",
    version="1.0.0"
)

# Зависимости
def get_session_manager():
    """Получение менеджера сессий"""
    return SessionManager()

class SessionManager:
    """Менеджер сессий интервью"""
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.SessionManager")
        self.agent_runtime = None  # Будет инициализирован позже
    
    async def create_session(self, request: CreateSessionRequest) -> SessionResponse:
        """Создание новой сессии интервью"""
        try:
            session_id = str(uuid.uuid4())
            room_name = f"interview-{session_id[:8]}"
            
            # Генерируем токены (заглушки для LiveKit)
            client_token = self._generate_jwt_token("client", room_name)
            agent_token = self._generate_jwt_token("agent", room_name)
            
            # Создаем сессию
            session_data = {
                "session_id": session_id,
                "room_name": room_name,
                "vacancy_id": request.vacancy_id,
                "resume_id": request.resume_id,
                "candidate_name": request.candidate_name,
                "interview_duration": request.interview_duration,
                "language": request.language,
                "status": "created",
                "created_at": datetime.now(),
                "updated_at": datetime.now(),
                "metrics": {
                    "total_duration": 0.0,
                    "speech_segments": 0,
                    "transcriptions": 0,
                    "emotions_detected": 0,
                    "questions_asked": 0,
                    "responses_generated": 0
                }
            }
            
            active_sessions[session_id] = session_data
            
            self.logger.info(f"Создана сессия {session_id} для кандидата {request.candidate_name}")
            
            return SessionResponse(
                session_id=session_id,
                room_name=room_name,
                livekit_url=settings.LIVEKIT_URL,
                client_token=client_token,
                agent_token=agent_token,
                status="created",
                created_at=session_data["created_at"],
                expires_at=session_data["created_at"] + timedelta(minutes=request.interview_duration + 10)
            )
            
        except Exception as e:
            self.logger.error(f"Ошибка создания сессии: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка создания сессии: {str(e)}")
    
    async def start_session(self, session_id: str) -> bool:
        """Запуск сессии (создание агента)"""
        try:
            if session_id not in active_sessions:
                raise HTTPException(status_code=404, detail="Сессия не найдена")
            
            session_data = active_sessions[session_id]
            
            if session_data["status"] != "created":
                raise HTTPException(status_code=400, detail="Сессия уже запущена или завершена")
            
            # Здесь будет запуск Agent-Runtime
            # Пока что просто меняем статус
            session_data["status"] = "running"
            session_data["updated_at"] = datetime.now()
            
            self.logger.info(f"Сессия {session_id} запущена")
            
            # TODO: Запустить Agent-Runtime
            # await self._start_agent_runtime(session_id, session_data)
            
            return True
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Ошибка запуска сессии {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка запуска сессии: {str(e)}")
    
    async def stop_session(self, session_id: str) -> StopSessionResponse:
        """Остановка сессии"""
        try:
            if session_id not in active_sessions:
                raise HTTPException(status_code=404, detail="Сессия не найдена")
            
            session_data = active_sessions[session_id]
            
            if session_data["status"] == "stopped":
                raise HTTPException(status_code=400, detail="Сессия уже остановлена")
            
            # Останавливаем агента
            # TODO: Остановить Agent-Runtime
            # await self._stop_agent_runtime(session_id)
            
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
    
    async def get_session_status(self, session_id: str) -> SessionStatusResponse:
        """Получение статуса сессии"""
        try:
            if session_id not in active_sessions:
                raise HTTPException(status_code=404, detail="Сессия не найдена")
            
            session_data = active_sessions[session_id]
            duration = (session_data["updated_at"] - session_data["created_at"]).total_seconds()
            
            return SessionStatusResponse(
                session_id=session_id,
                status=session_data["status"],
                duration=duration,
                metrics=session_data["metrics"],
                created_at=session_data["created_at"],
                updated_at=session_data["updated_at"]
            )
            
        except HTTPException:
            raise
        except Exception as e:
            self.logger.error(f"Ошибка получения статуса сессии {session_id}: {e}")
            raise HTTPException(status_code=500, detail=f"Ошибка получения статуса: {str(e)}")
    
    def _generate_jwt_token(self, role: str, room: str) -> str:
        """Генерация JWT токена (заглушка)"""
        # В реальной реализации здесь будет генерация настоящего JWT
        return f"jwt_token_{role}_{room}_{uuid.uuid4().hex[:16]}"

# API эндпоинты

@app.post("/api/sessions", response_model=SessionResponse)
async def create_session(
    request: CreateSessionRequest,
    background_tasks: BackgroundTasks,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Создание новой сессии интервью"""
    session = await session_manager.create_session(request)
    
    # Запускаем сессию в фоне
    background_tasks.add_task(session_manager.start_session, session.session_id)
    
    return session

@app.post("/api/sessions/{session_id}/stop", response_model=StopSessionResponse)
async def stop_session(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Остановка сессии интервью"""
    return await session_manager.stop_session(session_id)

@app.get("/api/sessions/{session_id}", response_model=SessionStatusResponse)
async def get_session_status(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Получение статуса сессии"""
    return await session_manager.get_session_status(session_id)

@app.get("/api/sessions/{session_id}/events")
async def get_session_events(session_id: str):
    """SSE поток событий сессии"""
    if session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    async def event_generator():
        """Генератор событий для SSE"""
        while True:
            if session_id not in active_sessions:
                break
                
            session_data = active_sessions[session_id]
            if session_data["status"] == "stopped":
                break
            
            # Отправляем текущие метрики
            yield f"data: {session_data['metrics']}\n\n"
            
            await asyncio.sleep(1)  # Обновляем каждую секунду
    
    return StreamingResponse(
        event_generator(),
        media_type="text/plain",
        headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
    )

# CRUD для вакансий
@app.post("/api/vacancies")
async def create_vacancy(request: VacancyRequest):
    """Создание вакансии"""
    vacancy_id = str(uuid.uuid4())
    vacancy_data = {
        "id": vacancy_id,
        "title": request.title,
        "description": request.description,
        "requirements": request.requirements,
        "skills": request.skills,
        "experience_level": request.experience_level,
        "salary_range": request.salary_range,
        "created_at": datetime.now()
    }
    
    vacancies[vacancy_id] = vacancy_data
    return {"id": vacancy_id, **vacancy_data}

@app.get("/api/vacancies/{vacancy_id}")
async def get_vacancy(vacancy_id: str):
    """Получение вакансии"""
    if vacancy_id not in vacancies:
        raise HTTPException(status_code=404, detail="Вакансия не найдена")
    return vacancies[vacancy_id]

@app.get("/api/vacancies")
async def list_vacancies():
    """Список всех вакансий"""
    return list(vacancies.values())

# CRUD для резюме
@app.post("/api/resumes")
async def create_resume(request: ResumeRequest):
    """Создание резюме"""
    resume_id = str(uuid.uuid4())
    resume_data = {
        "id": resume_id,
        "candidate_name": request.candidate_name,
        "email": request.email,
        "phone": request.phone,
        "experience": request.experience,
        "skills": request.skills,
        "education": request.education,
        "created_at": datetime.now()
    }
    
    resumes[resume_id] = resume_data
    return {"id": resume_id, **resume_data}

@app.get("/api/resumes/{resume_id}")
async def get_resume(resume_id: str):
    """Получение резюме"""
    if resume_id not in resumes:
        raise HTTPException(status_code=404, detail="Резюме не найдено")
    return resumes[resume_id]

@app.get("/api/resumes")
async def list_resumes():
    """Список всех резюме"""
    return list(resumes.values())

# Генерация отчетов
@app.post("/api/reports")
async def generate_report(request: ReportRequest):
    """Генерация отчета по сессии"""
    if request.session_id not in active_sessions:
        raise HTTPException(status_code=404, detail="Сессия не найдена")
    
    session_data = active_sessions[request.session_id]
    
    # TODO: Генерация реального отчета
    report = {
        "session_id": request.session_id,
        "candidate_name": session_data["candidate_name"],
        "duration": session_data["metrics"]["total_duration"],
        "transcript": "Транскрипт будет здесь...",
        "metrics": session_data["metrics"],
        "recommendation": "Рекомендация будет здесь...",
        "generated_at": datetime.now()
    }
    
    return report

@app.get("/api/health")
async def health_check():
    """Проверка здоровья сервиса"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "active_sessions": len(active_sessions),
        "total_vacancies": len(vacancies),
        "total_resumes": len(resumes)
    }

# Запуск приложения
if __name__ == "__main__":
    import uvicorn
    setup_logger(level="INFO")
    uvicorn.run(app, host="0.0.0.0", port=8000)
