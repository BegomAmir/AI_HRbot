"""
Утилиты для логирования
"""
import logging
import sys
from pathlib import Path
from typing import Optional
from src.config.settings import settings


def setup_logger(
    name: str = "ai_hr_bot",
    level: Optional[str] = None,
    log_file: Optional[str] = None
) -> logging.Logger:
    """
    Настройка логгера для приложения
    
    Args:
        name: Имя логгера
        level: Уровень логирования
        log_file: Путь к файлу логов
        
    Returns:
        Настроенный логгер
    """
    # Определяем уровень логирования
    log_level = level or settings.LOG_LEVEL
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Создаем логгер
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Очищаем существующие хендлеры
    logger.handlers.clear()
    
    # Создаем форматтер
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Хендлер для консоли
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Хендлер для файла (если указан)
    if log_file or settings.LOG_FILE:
        file_path = log_file or settings.LOG_FILE
        if file_path:
            # Создаем директорию для логов если не существует
            log_dir = Path(file_path).parent
            log_dir.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(file_path, encoding='utf-8')
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
    
    return logger


def get_logger(name: str = "ai_hr_bot") -> logging.Logger:
    """
    Получение настроенного логгера
    
    Args:
        name: Имя логгера
        
    Returns:
        Настроенный логгер
    """
    return logging.getLogger(name)


# Создаем основной логгер
main_logger = setup_logger("ai_hr_bot")
