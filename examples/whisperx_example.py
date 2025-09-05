#!/usr/bin/env python3
"""
Пример использования WhisperX STT сервиса
Демонстрирует возможности диаризации и выравнивания слов
"""
import asyncio
import sys
import os
import numpy as np

# Добавляем src в путь для импорта
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.whisperx_stt_service import WhisperXSTTService
from services.hybrid_stt_service import HybridSTTService
from models.interview import AudioSegment
from utils.logger import setup_logger, get_logger

logger = get_logger(__name__)


async def test_whisperx_direct():
    """Прямое тестирование WhisperX сервиса"""
    logger.info("=== Тестирование WhisperX STT сервиса ===")
    
    # Создаем WhisperX сервис
    whisperx_service = WhisperXSTTService()
    
    try:
        # Проверяем работоспособность
        is_healthy = await whisperx_service.health_check()
        if not is_healthy:
            logger.error("WhisperX сервис не работает")
            return
        
        logger.info("WhisperX сервис работает корректно")
        
        # Получаем информацию о модели
        model_info = whisperx_service.get_model_info()
        logger.info(f"Информация о модели: {model_info}")
        
        # Создаем тестовый аудио сегмент
        # В реальном использовании здесь был бы настоящий аудио файл
        test_audio = np.random.randn(16000).astype(np.float32) * 0.1  # 1 секунда
        test_segment = AudioSegment(
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            audio_data=test_audio.tobytes(),
            confidence=1.0,
            is_speech=True
        )
        
        # Тестируем базовую транскрипцию
        logger.info("Тестирование базовой транскрипции...")
        transcription = await whisperx_service.transcribe_audio(test_segment)
        
        if transcription:
            logger.info(f"✅ Транскрипция: '{transcription.text}'")
            logger.info(f"   Уверенность: {transcription.confidence:.2f}")
            logger.info(f"   Язык: {transcription.language}")
        else:
            logger.warning("❌ Транскрипция не удалась")
        
        # Тестируем транскрипцию с диаризацией
        logger.info("Тестирование транскрипции с диаризацией...")
        speaker_result = await whisperx_service.transcribe_with_speakers(test_segment)
        
        if speaker_result:
            logger.info("✅ Транскрипция с спикерами выполнена")
            logger.info(f"   Текст: '{speaker_result['text']}'")
            logger.info(f"   Сегменты спикеров: {len(speaker_result.get('speaker_segments', []))}")
            logger.info(f"   Сегменты слов: {len(speaker_result.get('word_segments', []))}")
            
            # Показываем детали спикеров
            for i, speaker_seg in enumerate(speaker_result.get('speaker_segments', [])[:3]):
                logger.info(f"   Спикер {i+1}: {speaker_seg.speaker} - '{speaker_seg.text[:50]}...'")
        else:
            logger.warning("❌ Транскрипция с спикерами не удалась")
        
        logger.info("Тестирование WhisperX завершено!")
        
    except Exception as e:
        logger.error(f"Ошибка тестирования WhisperX: {e}")


async def test_hybrid_stt():
    """Тестирование гибридного STT сервиса"""
    logger.info("=== Тестирование гибридного STT сервиса ===")
    
    # Создаем гибридный сервис
    hybrid_service = HybridSTTService()
    
    try:
        # Проверяем работоспособность
        is_healthy = await hybrid_service.health_check()
        if not is_healthy:
            logger.error("Гибридный STT сервис не работает")
            return
        
        logger.info("Гибридный STT сервис работает корректно")
        
        # Получаем информацию о сервисе
        model_info = hybrid_service.get_model_info()
        logger.info(f"Информация о сервисе: {model_info}")
        
        # Получаем возможности
        capabilities = hybrid_service.get_capabilities()
        logger.info(f"Возможности: {capabilities}")
        
        # Создаем тестовый аудио сегмент
        test_audio = np.random.randn(16000).astype(np.float32) * 0.1
        test_segment = AudioSegment(
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            audio_data=test_audio.tobytes(),
            confidence=1.0,
            is_speech=True
        )
        
        # Тестируем базовую транскрипцию
        logger.info("Тестирование базовой транскрипции...")
        transcription = await hybrid_service.transcribe_audio(test_segment)
        
        if transcription:
            logger.info(f"✅ Транскрипция: '{transcription.text}'")
            logger.info(f"   Уверенность: {transcription.confidence:.2f}")
        else:
            logger.warning("❌ Транскрипция не удалась")
        
        # Тестируем транскрипцию с продвинутыми функциями
        if capabilities.get('speaker_diarization', False):
            logger.info("Тестирование транскрипции с диаризацией...")
            speaker_result = await hybrid_service.transcribe_with_speakers(test_segment)
            
            if speaker_result:
                logger.info("✅ Транскрипция с спикерами выполнена")
                logger.info(f"   Текст: '{speaker_result['text']}'")
            else:
                logger.warning("❌ Транскрипция с спикерами не удалась")
        else:
            logger.info("ℹ️ Диаризация недоступна в текущей конфигурации")
        
        # Тестируем пакетную транскрипцию
        logger.info("Тестирование пакетной транскрипции...")
        test_segments = [test_segment, test_segment, test_segment]
        transcriptions = await hybrid_service.transcribe_batch(test_segments)
        
        logger.info(f"✅ Пакетная транскрипция: {len(transcriptions)} результатов")
        
        logger.info("Тестирование гибридного STT завершено!")
        
    except Exception as e:
        logger.error(f"Ошибка тестирования гибридного STT: {e}")


async def test_language_detection():
    """Тестирование определения языка"""
    logger.info("=== Тестирование определения языка ===")
    
    hybrid_service = HybridSTTService()
    
    try:
        # Создаем тестовые аудио сегменты
        test_audio = np.random.randn(16000).astype(np.float32) * 0.1
        test_segment = AudioSegment(
            start_time=0.0,
            end_time=1.0,
            duration=1.0,
            audio_data=test_audio.tobytes(),
            confidence=1.0,
            is_speech=True
        )
        
        # Тестируем определение языка
        if hybrid_service.get_capabilities().get('language_detection', False):
            logger.info("Тестирование автоопределения языка...")
            detected_language = hybrid_service.detect_language(test_segment)
            
            if detected_language:
                logger.info(f"✅ Определен язык: {detected_language}")
            else:
                logger.warning("❌ Не удалось определить язык")
        else:
            logger.info("ℹ️ Определение языка недоступно")
        
        # Получаем поддерживаемые языки
        supported_languages = hybrid_service.get_supported_languages()
        logger.info(f"Поддерживаемые языки: {len(supported_languages)}")
        logger.info(f"Примеры: {supported_languages[:10]}")
        
    except Exception as e:
        logger.error(f"Ошибка тестирования определения языка: {e}")


async def main():
    """Главная функция тестирования"""
    # Настройка логирования
    setup_logger(level="INFO")
    
    try:
        # Тестируем WhisperX напрямую
        await test_whisperx_direct()
        
        print("\n" + "="*50 + "\n")
        
        # Тестируем гибридный сервис
        await test_hybrid_stt()
        
        print("\n" + "="*50 + "\n")
        
        # Тестируем определение языка
        await test_language_detection()
        
        logger.info("Все тесты завершены!")
        
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())
