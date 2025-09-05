# Установка и настройка Python Agent для AI HR Bot

## Требования

- Python 3.8+
- pip
- Доступ к интернету для загрузки моделей

## Установка

### 1. Клонирование репозитория

```bash
git clone <your-repo-url>
cd AI_HRbot
```

### 2. Создание виртуального окружения

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# или
venv\Scripts\activate  # Windows
```

### 3. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 4. Настройка переменных окружения

```bash
cp env.example .env
# Отредактируйте .env файл с вашими настройками
```

### 5. Загрузка языковых моделей

```bash
# Для русского языка
python -m spacy download ru_core_news_sm

# Для английского языка
python -m spacy download en_core_web_sm
```

## Запуск

### Основной агент

```bash
python src/main.py
```

### Тестирование

```bash
python test_agent.py
```

### Примеры использования

```bash
# Тестирование TTS сервиса
python examples/tts_example.py

# Тестирование WhisperX STT с диаризацией
python examples/whisperx_example.py
```

## Структура проекта

```
AI_HRbot/
├── src/
│   ├── services/           # Микросервисы
│   │   ├── audio_processing.py    # Обработка аудио
│   │   ├── vad_service.py         # Voice Activity Detection
│   │   ├── stt_service.py         # Speech-to-Text (faster-whisper)
│   │   ├── whisperx_stt_service.py # STT с WhisperX (диаризация)
│   │   ├── hybrid_stt_service.py  # Гибридный STT сервис
│   │   ├── prosody_service.py     # Анализ просодии
│   │   ├── emotion_service.py     # Распознавание эмоций
│   │   └── tts_service.py         # Text-to-Speech
│   ├── models/            # Модели данных
│   ├── config/            # Конфигурация
│   ├── utils/             # Утилиты
│   └── main.py           # Главный файл
├── requirements.txt       # Зависимости
├── env.example           # Пример переменных окружения
├── test_agent.py         # Тестовый скрипт
└── examples/             # Примеры использования
    ├── tts_example.py    # Пример TTS сервиса
    └── whisperx_example.py # Пример WhisperX STT
```

## Конфигурация

### Основные настройки

- `LIVEKIT_URL` - URL LiveKit сервера
- `STT_MODEL_SIZE` - Размер модели Whisper (tiny, base, small, medium, large)
- `STT_LANGUAGE` - Язык для STT (ru, en, auto)
- `VAD_THRESHOLD` - Порог для VAD (0.0-1.0)

### Настройка моделей

#### STT (Speech-to-Text)
- **faster-whisper**: быстрая транскрипция
- **WhisperX**: высокая точность + диаризация + выравнивание слов
- **Гибридный режим**: автоматический выбор лучшего движка
- Размеры моделей: `tiny`, `base`, `small`, `medium`, `large`

#### VAD
- `threshold` - чувствительность обнаружения речи
- `min_speech_duration` - минимальная длительность речи
- `max_speech_duration` - максимальная длительность речи

#### TTS
- `TTS_VOICE` - голос для синтеза речи
- `TTS_SPEED` - скорость речи (0.5-2.0)

## Устранение неполадок

### Ошибка загрузки моделей

```bash
# Очистка кэша
pip cache purge

# Переустановка зависимостей
pip uninstall -r requirements.txt
pip install -r requirements.txt
```

### Проблемы с аудио

- Убедитесь, что установлены системные аудио библиотеки
- На Linux: `sudo apt-get install portaudio19-dev`
- На Mac: `brew install portaudio`

### Проблемы с CUDA

Если у вас есть GPU NVIDIA:

```bash
# Установка PyTorch с CUDA
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## Разработка

### Добавление нового сервиса

1. Создайте файл в `src/services/`
2. Реализуйте интерфейс с методами:
   - `__init__()` - инициализация
   - `health_check()` - проверка работоспособности
3. Добавьте сервис в `AIHRAgent` в `src/main.py`

### Логирование

```python
from src.utils.logger import get_logger

logger = get_logger(__name__)
logger.info("Информационное сообщение")
logger.error("Ошибка")
logger.debug("Отладочная информация")
```

## Производительность

### Оптимизация для CPU

- Используйте `STT_MODEL_SIZE=tiny` для быстрой работы
- Уменьшите `VAD_THRESHOLD` для менее чувствительного VAD
- Настройте размеры буферов в `AudioBuffer`

### Оптимизация для GPU

- Установите PyTorch с CUDA
- Измените `device="cuda"` в STT сервисе
- Используйте `compute_type="float16"` для экономии памяти

## Мониторинг

### Health Check

```python
status = await agent.get_status()
print(f"Агент работает: {status['is_running']}")
print(f"Длительность буфера: {status['buffer_duration']:.2f}s")
```

### Логи

Логи сохраняются в файл, указанный в `LOG_FILE` переменной окружения.

## Возможности WhisperX

### Диаризация (Speaker Diarization)
- Автоматическое разделение речи разных спикеров
- Определение временных меток для каждого спикера
- Поддержка множественных участников интервью

### Выравнивание слов (Word Alignment)
- Точные временные метки для каждого слова
- Улучшенная точность транскрипции
- Поддержка длинных аудио без потери качества

### Гибридный режим
- Автоматический выбор между WhisperX и faster-whisper
- Fallback на faster-whisper если WhisperX недоступен
- Оптимальная производительность в любых условиях

### Пример использования

```python
from src.main import AIHRAgent

agent = AIHRAgent()
await agent.start()

# Транскрипция с диаризацией
speaker_result = await agent.transcribe_with_speakers(audio_segment)

# Получение возможностей STT
capabilities = agent.get_stt_capabilities()
print(f"Диаризация доступна: {capabilities['speaker_diarization']}")
```

## Интеграция

### С LiveKit

```python
# Подключение к LiveKit комнате
# (реализация зависит от вашего LiveKit клиента)
```

### С Django API

```python
# Отправка метрик в Django
# (реализация зависит от вашего Django API)
```

## Поддержка

При возникновении проблем:

1. Проверьте логи
2. Убедитесь, что все зависимости установлены
3. Проверьте настройки в `.env` файле
4. Запустите тесты: `python test_agent.py`
