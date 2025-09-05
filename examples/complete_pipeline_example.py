#!/usr/bin/env python3
"""
Пример полного пайплайна AI HR Agent
Демонстрирует работу всех сервисов в последовательности
"""
import asyncio
import sys
import os
import numpy as np

# Добавляем src в путь для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import AIHRAgent
from models.interview import AudioSegment
from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


async def test_complete_pipeline():
    """Тестирование полного пайплайна"""
    logger.info("=== Тестирование полного пайплайна AI HR Agent ===")
    
    # Создаем агента
    agent = AIHRAgent()
    
    try:
        # Запускаем агента
        await agent.start()
        logger.info("✅ Агент запущен")
        
        # Симулируем входящий аудио поток
        logger.info("📡 Симуляция входящего аудио потока...")
        
        for i in range(5):
            # Создаем тестовые аудио данные
            sample_rate = 16000
            duration = 0.5  # 0.5 секунды
            frequency = 440 + i * 100  # Разные частоты
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            # Конвертируем в байты
            audio_bytes = audio_data.tobytes()
            
            # Отправляем в агента
            await agent.process_audio_stream(audio_bytes, timestamp=i * 0.5)
            
            # Ждем немного
            await asyncio.sleep(0.1)
        
        # Ждем обработки
        await asyncio.sleep(2)
        
        # Тестируем endpointing
        logger.info("🎯 Тестирование endpointing...")
        endpointing_status = agent.get_endpointing_status()
        logger.info(f"Статус endpointing: {endpointing_status}")
        
        # Принудительно завершаем речь
        endpoint_result = await agent.force_speech_endpoint()
        if endpoint_result:
            logger.info(f"✅ Речь завершена: {endpoint_result['duration']:.2f}s")
        
        # Тестируем генерацию ответа бота
        logger.info("🤖 Тестирование генерации ответа бота...")
        
        test_responses = [
            "Привет! Меня зовут Оксана, я буду проводить интервью.",
            "Расскажите о вашем опыте работы с Python.",
            "Спасибо за ответ, следующий вопрос."
        ]
        
        for i, response in enumerate(test_responses):
            logger.info(f"Генерация ответа {i+1}: {response[:30]}...")
            
            # Генерируем ответ с разными эмоциями
            emotions = ["neutral", "happy", "neutral"]
            emotion = emotions[i] if i < len(emotions) else "neutral"
            
            audio_response = await agent.generate_bot_response(response, emotion)
            
            if audio_response:
                logger.info(f"✅ Ответ {i+1} сгенерирован: {audio_response.duration:.2f}s")
            else:
                logger.error(f"❌ Ошибка генерации ответа {i+1}")
            
            await asyncio.sleep(0.5)
        
        # Получаем статистику
        logger.info("📊 Получение статистики...")
        
        # Статистика publisher
        publisher_stats = agent.get_publisher_statistics()
        logger.info(f"Статистика publisher: {publisher_stats}")
        
        # Статистика LLM
        llm_stats = agent.get_llm_statistics()
        logger.info(f"Статистика LLM: {llm_stats}")
        
        # Возможности STT
        stt_capabilities = agent.get_stt_capabilities()
        logger.info(f"Возможности STT: {stt_capabilities}")
        
        # Общий статус
        status = await agent.get_status()
        logger.info(f"Общий статус агента: {status}")
        
        # Проверяем созданные файлы
        logger.info("📁 Проверка созданных файлов...")
        
        # Проверяем директорию LLM данных
        llm_dir = "llm_data"
        if os.path.exists(llm_dir):
            llm_files = os.listdir(llm_dir)
            logger.info(f"Файлы LLM данных: {len(llm_files)}")
            for file in llm_files[:3]:  # Показываем первые 3 файла
                logger.info(f"  - {file}")
        
        # Проверяем директорию output (если включена)
        output_dir = "output"
        if os.path.exists(output_dir):
            output_files = os.listdir(output_dir)
            logger.info(f"Файлы output: {len(output_files)}")
            for file in output_files[:3]:  # Показываем первые 3 файла
                logger.info(f"  - {file}")
        
        logger.info("🎉 Полный пайплайн протестирован успешно!")
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования: {e}")
        
    finally:
        # Останавливаем агента
        await agent.stop()
        logger.info("🛑 Агент остановлен")


async def test_individual_services():
    """Тестирование отдельных сервисов"""
    logger.info("=== Тестирование отдельных сервисов ===")
    
    agent = AIHRAgent()
    
    try:
        await agent.start()
        
        # Тестируем VAD
        logger.info("🔍 Тестирование VAD...")
        vad_healthy = await agent.vad_service.health_check()
        logger.info(f"VAD здоров: {vad_healthy}")
        
        # Тестируем STT
        logger.info("🎤 Тестирование STT...")
        stt_healthy = await agent.stt_service.health_check()
        logger.info(f"STT здоров: {stt_healthy}")
        
        # Тестируем TTS
        logger.info("🔊 Тестирование TTS...")
        tts_healthy = await agent.tts_service.health_check()
        logger.info(f"TTS здоров: {tts_healthy}")
        
        # Тестируем Publisher
        logger.info("📤 Тестирование Publisher...")
        publisher_healthy = await agent.publisher_service.health_check()
        logger.info(f"Publisher здоров: {publisher_healthy}")
        
        # Тестируем LLM Integration
        logger.info("🧠 Тестирование LLM Integration...")
        llm_healthy = await agent.llm_integration_service.health_check()
        logger.info(f"LLM Integration здоров: {llm_healthy}")
        
        logger.info("✅ Все сервисы протестированы")
        
    except Exception as e:
        logger.error(f"❌ Ошибка тестирования сервисов: {e}")
        
    finally:
        await agent.stop()


async def main():
    """Главная функция"""
    # Настройка логирования
    setup_logger(level="INFO")
    
    try:
        # Тестируем отдельные сервисы
        await test_individual_services()
        
        print("\n" + "="*60 + "\n")
        
        # Тестируем полный пайплайн
        await test_complete_pipeline()
        
        logger.info("🎯 Все тесты завершены успешно!")
        
    except Exception as e:
        logger.error(f"💥 Критическая ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
