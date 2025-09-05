#!/usr/bin/env python3
"""
Пример клиента для тестирования FastAPI сервиса
"""
import asyncio
import aiohttp
import json
from typing import Dict, Any

BASE_URL = "http://localhost:8000"


class AIHRBotClient:
    """Клиент для AI HR Bot API"""
    
    def __init__(self, base_url: str = BASE_URL):
        self.base_url = base_url
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    async def health_check(self) -> Dict[str, Any]:
        """Проверка здоровья сервиса"""
        async with self.session.get(f"{self.base_url}/api/health") as response:
            return await response.json()
    
    async def create_vacancy(self, title: str, description: str, requirements: list, skills: list) -> Dict[str, Any]:
        """Создание вакансии"""
        data = {
            "title": title,
            "description": description,
            "requirements": requirements,
            "skills": skills,
            "experience_level": "middle"
        }
        
        async with self.session.post(f"{self.base_url}/api/vacancies", json=data) as response:
            return await response.json()
    
    async def create_resume(self, candidate_name: str, email: str, experience: list, skills: list) -> Dict[str, Any]:
        """Создание резюме"""
        data = {
            "candidate_name": candidate_name,
            "email": email,
            "experience": experience,
            "skills": skills,
            "education": []
        }
        
        async with self.session.post(f"{self.base_url}/api/resumes", json=data) as response:
            return await response.json()
    
    async def create_session(self, vacancy_id: str, resume_id: str, candidate_name: str) -> Dict[str, Any]:
        """Создание сессии интервью"""
        data = {
            "vacancy_id": vacancy_id,
            "resume_id": resume_id,
            "candidate_name": candidate_name,
            "interview_duration": 30,
            "language": "ru"
        }
        
        async with self.session.post(f"{self.base_url}/api/sessions", json=data) as response:
            return await response.json()
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Получение статуса сессии"""
        async with self.session.get(f"{self.base_url}/api/sessions/{session_id}") as response:
            return await response.json()
    
    async def stop_session(self, session_id: str) -> Dict[str, Any]:
        """Остановка сессии"""
        async with self.session.post(f"{self.base_url}/api/sessions/{session_id}/stop") as response:
            return await response.json()
    
    async def generate_bot_response(self, session_id: str, response_text: str, emotion: str = "neutral") -> Dict[str, Any]:
        """Генерация ответа бота"""
        params = {
            "response_text": response_text,
            "emotion": emotion
        }
        
        async with self.session.post(f"{self.base_url}/api/runtime/{session_id}/response", params=params) as response:
            return await response.json()
    
    async def get_runtime_status(self) -> Dict[str, Any]:
        """Получение статуса runtime"""
        async with self.session.get(f"{self.base_url}/api/runtime/status") as response:
            return await response.json()


async def test_api():
    """Тестирование API"""
    print("🚀 Тестирование AI HR Bot API")
    
    async with AIHRBotClient() as client:
        try:
            # 1. Проверка здоровья
            print("\n1. Проверка здоровья сервиса...")
            health = await client.health_check()
            print(f"✅ Статус: {health['status']}")
            print(f"   Активных сессий: {health.get('active_sessions', 0)}")
            
            # 2. Создание вакансии
            print("\n2. Создание вакансии...")
            vacancy = await client.create_vacancy(
                title="Python Developer",
                description="Разработка веб-приложений на Python",
                requirements=["Python 3.8+", "FastAPI", "PostgreSQL"],
                skills=["Python", "FastAPI", "SQL", "Docker"]
            )
            vacancy_id = vacancy["id"]
            print(f"✅ Вакансия создана: {vacancy_id}")
            
            # 3. Создание резюме
            print("\n3. Создание резюме...")
            resume = await client.create_resume(
                candidate_name="Иван Петров",
                email="ivan@example.com",
                experience=[
                    {"company": "TechCorp", "position": "Python Developer", "duration": "2 years"},
                    {"company": "StartupXYZ", "position": "Backend Developer", "duration": "1 year"}
                ],
                skills=["Python", "FastAPI", "Django", "PostgreSQL", "Redis"]
            )
            resume_id = resume["id"]
            print(f"✅ Резюме создано: {resume_id}")
            
            # 4. Создание сессии
            print("\n4. Создание сессии интервью...")
            session = await client.create_session(vacancy_id, resume_id, "Иван Петров")
            session_id = session["session_id"]
            print(f"✅ Сессия создана: {session_id}")
            print(f"   Комната: {session['room_name']}")
            print(f"   LiveKit URL: {session['livekit_url']}")
            
            # 5. Проверка статуса сессии
            print("\n5. Проверка статуса сессии...")
            await asyncio.sleep(2)  # Ждем запуска
            status = await client.get_session_status(session_id)
            print(f"✅ Статус сессии: {status['status']}")
            print(f"   Длительность: {status['duration']:.2f}s")
            
            # 6. Проверка статуса runtime
            print("\n6. Проверка статуса runtime...")
            runtime_status = await client.get_runtime_status()
            print(f"✅ Runtime статус:")
            print(f"   Всего runtime: {runtime_status['total_runtimes']}")
            print(f"   Активных: {runtime_status['active_runtimes']}")
            
            # 7. Генерация ответа бота
            print("\n7. Генерация ответа бота...")
            bot_response = await client.generate_bot_response(
                session_id, 
                "Привет! Меня зовут Оксана, я буду проводить интервью.",
                "happy"
            )
            print(f"✅ Ответ бота: {bot_response['message']}")
            print(f"   Длительность аудио: {bot_response['audio_duration']:.2f}s")
            
            # 8. Остановка сессии
            print("\n8. Остановка сессии...")
            stop_result = await client.stop_session(session_id)
            print(f"✅ Сессия остановлена: {stop_result['status']}")
            print(f"   Финальная длительность: {stop_result['final_metrics']['total_duration']:.2f}s")
            
            print("\n🎉 Все тесты пройдены успешно!")
            
        except Exception as e:
            print(f"❌ Ошибка тестирования: {e}")


async def test_sse_events():
    """Тестирование SSE событий"""
    print("\n📡 Тестирование SSE событий...")
    
    async with aiohttp.ClientSession() as session:
        try:
            # Создаем сессию для тестирования SSE
            async with AIHRBotClient() as client:
                # Создаем тестовую сессию
                vacancy = await client.create_vacancy("Test", "Test", [], [])
                resume = await client.create_resume("Test", "test@test.com", [], [])
                session_data = await client.create_session(vacancy["id"], resume["id"], "Test")
                session_id = session_data["session_id"]
                
                print(f"📡 Подключение к SSE потоку для сессии {session_id}...")
                
                # Подключаемся к SSE потоку
                async with session.get(f"{BASE_URL}/api/sessions/{session_id}/events") as response:
                    if response.status == 200:
                        print("✅ SSE подключение установлено")
                        
                        # Читаем события в течение 10 секунд
                        timeout = 10
                        start_time = asyncio.get_event_loop().time()
                        
                        async for line in response.content:
                            if asyncio.get_event_loop().time() - start_time > timeout:
                                break
                            
                            if line.startswith(b'data: '):
                                data = line[6:].decode('utf-8').strip()
                                if data:
                                    try:
                                        event_data = json.loads(data)
                                        print(f"📊 Событие: {event_data}")
                                    except json.JSONDecodeError:
                                        print(f"📊 Сырые данные: {data}")
                        
                        # Останавливаем тестовую сессию
                        await client.stop_session(session_id)
                        print("✅ SSE тест завершен")
                    else:
                        print(f"❌ Ошибка SSE подключения: {response.status}")
                        
        except Exception as e:
            print(f"❌ Ошибка SSE тестирования: {e}")


async def main():
    """Главная функция"""
    print("🤖 AI HR Bot API Client Test")
    print("=" * 50)
    
    # Основные тесты API
    await test_api()
    
    # Тест SSE событий
    await test_sse_events()
    
    print("\n🏁 Тестирование завершено!")


if __name__ == "__main__":
    asyncio.run(main())
