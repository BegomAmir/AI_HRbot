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

# Тестирование полного пайплайна
python examples/complete_pipeline_example.py
```

## Структура проекта

```
AI_HRbot/
├── src/
│   ├── services/           # Микросервисы
│   │   ├── audio_processing.py    # Обработка аудио
│   │   ├── vad_service.py         # Voice Activity Detection
│   │   ├── endpointing_service.py # Определение конца речи
│   │   ├── stt_service.py         # Speech-to-Text (faster-whisper)
│   │   ├── whisperx_stt_service.py # STT с WhisperX (диаризация)
│   │   ├── hybrid_stt_service.py  # Гибридный STT сервис
│   │   ├── prosody_service.py     # Анализ просодии
│   │   ├── emotion_service.py     # Распознавание эмоций
│   │   ├── tts_service.py         # Text-to-Speech
│   │   ├── publisher_service.py   # Публикация аудио бота
│   │   └── llm_integration_service.py # Интеграция с LLM
│   ├── models/            # Модели данных
│   ├── config/            # Конфигурация
│   ├── utils/             # Утилиты
│   └── main.py           # Главный файл
├── requirements.txt       # Зависимости
├── env.example           # Пример переменных окружения
├── test_agent.py         # Тестовый скрипт
└── examples/             # Примеры использования
    ├── tts_example.py    # Пример TTS сервиса
    ├── whisperx_example.py # Пример WhisperX STT
    └── complete_pipeline_example.py # Полный пайплайн
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

# Принудительное завершение речи
endpoint = await agent.force_speech_endpoint()

# Получение статистики
publisher_stats = agent.get_publisher_statistics()
llm_stats = agent.get_llm_statistics()
```

## Новые сервисы

### Endpointing Service
- **Определение конца речи** на основе VAD и временных меток
- **Настраиваемые параметры**: порог тишины, минимальная/максимальная длительность
- **Принудительное завершение** речи по команде
- **Мониторинг состояния** активной речи

### Publisher Service
- **Публикация аудио бота** в различные системы
- **LiveKit интеграция** для реального времени
- **Файловый вывод** для отладки и архивирования
- **Webhook поддержка** для внешних систем
- **Метрики и статистика** публикации

### LLM Integration Service
- **Подготовка данных** для отправки в LLM
- **Форматирование текста** и метаданных
- **Сохранение в файлы** для внешней обработки
- **Агрегация данных** от всех сервисов
- **Обработка ответов** от LLM

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

## FastAPI Service

### Архитектура

Сервис состоит из двух основных компонентов:

1. **Gateway** - REST API для управления сессиями
2. **Agent Runtime** - управление ботом-участником в LiveKit

### API Endpoints

#### Сессии интервью
- `POST /api/sessions` - создание сессии
- `POST /api/sessions/{id}/stop` - остановка сессии  
- `GET /api/sessions/{id}` - статус сессии
- `GET /api/sessions/{id}/events` - SSE поток событий

#### Управление данными
- `POST /api/vacancies` - создание вакансии
- `GET /api/vacancies` - список вакансий
- `POST /api/resumes` - создание резюме
- `GET /api/resumes` - список резюме

#### Agent Runtime
- `GET /api/runtime/status` - статус всех runtime
- `GET /api/runtime/{id}/status` - статус конкретного runtime
- `POST /api/runtime/{id}/response` - генерация ответа бота

#### Отчеты
- `POST /api/reports` - генерация отчета по сессии

### Переменные окружения

#### Основные настройки
```bash
# FastAPI
PYTHONPATH=/app
PYTHONUNBUFFERED=1

# LiveKit
LIVEKIT_URL=ws://localhost:7880
LIVEKIT_API_KEY=your_api_key
LIVEKIT_API_SECRET=your_api_secret

# STT
STT_MODEL_SIZE=base
STT_LANGUAGE=ru

# VAD
VAD_THRESHOLD=0.5

# TTS
TTS_VOICE=ru_ru-oksana-medium
TTS_SPEED=1.0

# Логирование
LOG_LEVEL=INFO
LOG_FILE=logs/ai_hr_bot.log
```

### Интеграционные точки

#### Входящие данные (Input)
1. **RTP Audio Stream** → `audio_processing.py`
   - Переменная: `AUDIO_SAMPLE_RATE`, `AUDIO_CHUNK_SIZE`
   - Метод: `process_audio_chunk()`

2. **Session Creation** → `gateway.py`
   - Endpoint: `POST /api/sessions`
   - Данные: vacancy_id, resume_id, candidate_name

3. **LiveKit Audio** → `agent_runtime.py`
   - Переменная: `LIVEKIT_URL`, `LIVEKIT_API_KEY`
   - Метод: `process_audio_chunk()`

#### Исходящие данные (Output)
1. **LLM Data** → `llm_integration_service.py`
   - Директория: `llm_data/`
   - Формат: `.txt` файлы с транскрипциями и метаданными

2. **Bot Audio** → `publisher_service.py`
   - Директория: `output/`
   - Формат: `.wav` файлы + `.json` метаданные

3. **Telemetry** → `agent_runtime.py`
   - Endpoint: SSE `GET /api/sessions/{id}/events`
   - Формат: JSON события в реальном времени

4. **Reports** → `gateway.py`
   - Endpoint: `POST /api/reports`
   - Формат: JSON отчет с рекомендациями

### Запуск сервиса

#### Локальная разработка
```bash
# Установка зависимостей
pip install -r requirements.txt

# Запуск сервиса
python -m src.api.main

# Или с uvicorn
uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Docker
```bash
# Сборка и запуск
docker-compose up --build

# Только сервис
docker build -t ai-hr-bot .
docker run -p 8000:8000 ai-hr-bot
```

#### Тестирование API
```bash
# Запуск тестового клиента
python examples/api_client_example.py

# Проверка здоровья
curl http://localhost:8000/api/health
```

### Мониторинг

#### Health Check
- Endpoint: `GET /api/health`
- Проверяет: Gateway, Agent Runtime, LiveKit, LLM

#### Метрики
- Активные сессии: `active_sessions`
- Runtime статистика: `total_runtimes`, `active_runtimes`
- Обработанные аудио чанки: `audio_chunks_processed`
- Сгенерированные ответы: `bot_responses_generated`

#### Логи
- Файл: `logs/ai_hr_bot.log`
- Уровень: настраивается через `LOG_LEVEL`
- Формат: JSON с временными метками

## Поддержка

При возникновении проблем:

1. Проверьте логи
2. Убедитесь, что все зависимости установлены
3. Проверьте настройки в `.env` файле
4. Запустите тесты: `python test_agent.py`
