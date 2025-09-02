#!/usr/bin/env python3
"""
Пример использования TTS сервиса
"""
import asyncio
import sys
import os

# Добавляем src в путь для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.tts_service import TTSService
from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


async def test_tts():
    """Тестирование TTS сервиса"""
    logger.info("Запуск тестирования TTS сервиса...")
    
    # Создаем TTS сервис
    tts = TTSService()
    
    try:
        # Проверяем работоспособность
        is_healthy = await tts.health_check()
        if not is_healthy:
            logger.error("TTS сервис не работает")
            return
        
        logger.info("TTS сервис работает корректно")
        
        # Получаем информацию о доступных голосах
        voices = tts.get_available_voices()
        logger.info(f"Доступные голоса: {voices}")
        
        # Тестируем синтез речи
        test_texts = [
            "Привет! Меня зовут Оксана.",
            "Я буду проводить интервью.",
            "Расскажите о вашем опыте работы."
        ]
        
        for i, text in enumerate(test_texts):
            logger.info(f"Синтез текста {i+1}: {text}")
            
            # Синтезируем речь
            audio_segment = await tts.synthesize_speech(text)
            
            if audio_segment:
                logger.info(f"✅ Аудио сгенерировано: {audio_segment.duration:.2f}s")
                logger.info(f"   Размер данных: {len(audio_segment.audio_data)} байт")
            else:
                logger.error(f"❌ Ошибка генерации аудио для текста {i+1}")
        
        # Тестируем синтез с эмоциями
        logger.info("Тестирование синтеза с эмоциями...")
        
        emotion_texts = {
            "happy": "Отлично! Вы очень хорошо справились с задачей!",
            "sad": "К сожалению, ваш ответ не соответствует требованиям.",
            "neutral": "Спасибо за ответ. Следующий вопрос."
        }
        
        for emotion, text in emotion_texts.items():
            logger.info(f"Синтез с эмоцией '{emotion}': {text}")
            
            audio_segment = await tts.synthesize_interview_response(text, emotion)
            
            if audio_segment:
                logger.info(f"✅ Эмоциональный ответ сгенерирован: {audio_segment.duration:.2f}s")
            else:
                logger.error(f"❌ Ошибка генерации эмоционального ответа")
        
        # Тестируем пакетный синтез
        logger.info("Тестирование пакетного синтеза...")
        
        batch_texts = [
            "Первый вопрос.",
            "Второй вопрос.",
            "Третий вопрос."
        ]
        
        audio_segments = await tts.synthesize_batch(batch_texts)
        logger.info(f"✅ Пакетный синтез: {len(audio_segments)} аудио сегментов")
        
        # Получаем информацию о голосе
        voice_info = tts.get_voice_info("ru_ru-oksana-medium")
        logger.info(f"Информация о голосе: {voice_info}")
        
        logger.info("Тестирование TTS завершено успешно!")
        
    except Exception as e:
        logger.error(f"Ошибка тестирования TTS: {e}")


async def main():
    """Главная функция"""
    # Настройка логирования
    setup_logger(level="INFO")
    
    try:
        await test_tts()
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
