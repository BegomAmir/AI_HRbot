#!/usr/bin/env python3
"""
Тестовый скрипт для проверки работы Python Agent'а
"""
import asyncio
import numpy as np
from src.main import AIHRAgent
from src.utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


async def test_audio_callback(audio_segment):
    """Тестовый колбэк для аудио событий"""
    logger.info(f"Получен аудио сегмент: {audio_segment.duration:.2f}s")


async def test_analysis_callback(analysis_result):
    """Тестовый колбэк для результатов анализа"""
    logger.info("Получены результаты анализа:")
    logger.info(f"  Транскрипции: {len(analysis_result['transcriptions'])}")
    logger.info(f"  Просодия: {len(analysis_result['prosody'])}")
    logger.info(f"  Эмоции: {len(analysis_result['emotions'])}")


async def test_tts_callback(audio_segment):
    """Тестовый колбэк для TTS ответов"""
    logger.info(f"Получен TTS ответ: {audio_segment.duration:.2f}s")


async def test_agent():
    """Тестирование агента"""
    logger.info("Запуск тестирования AI HR Agent...")
    
    # Создаем агента
    agent = AIHRAgent()
    
    # Добавляем колбэки
    agent.add_audio_callback(test_audio_callback)
    agent.add_analysis_callback(test_analysis_callback)
    agent.add_audio_callback(test_tts_callback)  # Для TTS ответов
    
    try:
        # Запускаем агента
        await agent.start()
        
        # Симулируем входящий аудио поток
        logger.info("Симуляция входящего аудио потока...")
        
        for i in range(3):
            # Создаем тестовые аудио данные (синусоида)
            sample_rate = 16000
            duration = 0.5  # 0.5 секунды
            frequency = 440  # 440 Hz (нота A)
            
            t = np.linspace(0, duration, int(sample_rate * duration), False)
            audio_data = np.sin(2 * np.pi * frequency * t).astype(np.float32)
            
            # Конвертируем в байты
            audio_bytes = audio_data.tobytes()
            
            # Отправляем в агента
            await agent.process_audio_stream(audio_bytes, timestamp=i * 0.5)
            
            # Ждем немного
            await asyncio.sleep(1)
        
        # Тестируем TTS
        logger.info("Тестирование TTS сервиса...")
        
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
        
        # Тестируем возможности STT
        logger.info("Тестирование возможностей STT...")
        capabilities = agent.get_stt_capabilities()
        logger.info(f"Возможности STT: {capabilities}")
        
        # Тестируем транскрипцию с диаризацией (если доступно)
        if capabilities.get('speaker_diarization', False):
            logger.info("Тестирование транскрипции с диаризацией...")
            
            # Создаем тестовый аудио сегмент для диаризации
            test_audio = np.random.randn(32000).astype(np.float32) * 0.1  # 2 секунды
            test_segment = AudioSegment(
                start_time=0.0,
                end_time=2.0,
                duration=2.0,
                audio_data=test_audio.tobytes(),
                confidence=1.0,
                is_speech=True
            )
            
            speaker_result = await agent.transcribe_with_speakers(test_segment)
            if speaker_result:
                logger.info(f"✅ Диаризация выполнена: {len(speaker_result.get('speaker_segments', []))} сегментов")
            else:
                logger.warning("❌ Диаризация не удалась")
        else:
            logger.info("ℹ️ Диаризация недоступна")
        
        # Получаем статус
        status = await agent.get_status()
        logger.info(f"Статус агента: {status}")
        
        # Останавливаем агента
        await agent.stop()
        
    except Exception as e:
        logger.error(f"Ошибка тестирования: {e}")
        await agent.stop()


async def main():
    """Главная функция тестирования"""
    # Настройка логирования
    setup_logger(level="DEBUG")
    
    try:
        await test_agent()
        logger.info("Тестирование завершено успешно")
    except Exception as e:
        logger.error(f"Ошибка тестирования: {e}")


if __name__ == "__main__":
    asyncio.run(main())
