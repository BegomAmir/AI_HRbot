"""
Сервис распознавания эмоций в речи (Speech Emotion Recognition)
Анализирует эмоциональную окраску голоса
"""
import asyncio
import numpy as np
import librosa
from typing import List, Optional, Dict, Any, Tuple
from src.models.interview import AudioSegment, EmotionAnalysis
from src.utils.logger import get_logger
from src.config.settings import settings

logger = get_logger(__name__)


class EmotionRecognitionService:
    """
    Сервис распознавания эмоций в речи
    
    Анализирует:
    - Основную эмоцию
    - Распределение эмоций
    - Валентность (позитив/негатив)
    - Возбуждение (активность)
    """
    
    def __init__(self):
        self.logger = get_logger(f"{__name__}.EmotionRecognitionService")
        self.sample_rate = 16000
        
        # Базовые эмоции для анализа
        self.emotions = [
            "neutral", "happy", "sad", "angry", "fearful", "disgusted", "surprised"
        ]
        
        # Инициализация модели (если доступна)
        self.model = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Инициализация модели распознавания эмоций"""
        try:
            # Здесь можно загрузить предобученную модель
            # Например, Wav2Vec2 для эмоций или другую
            # Пока используем простые акустические признаки
            self.logger.info("Используется акустический анализ эмоций")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации модели эмоций: {e}")
    
    async def recognize_emotion(
        self, 
        audio_segment: AudioSegment
    ) -> Optional[EmotionAnalysis]:
        """
        Распознавание эмоций в аудио сегменте
        
        Args:
            audio_segment: Аудио сегмент для анализа
            
        Returns:
            Анализ эмоций или None при ошибке
        """
        try:
            # Конвертируем аудио в numpy array
            audio_array = np.frombuffer(audio_segment.audio_data, dtype=np.float32)
            
            if len(audio_array) == 0:
                self.logger.warning("Пустой аудио сегмент для анализа эмоций")
                return None
            
            # Извлекаем акустические признаки
            features = await self._extract_acoustic_features(audio_array)
            
            # Анализируем эмоции на основе признаков
            emotion_result = await self._analyze_emotions_from_features(features)
            
            # Вычисляем валентность и возбуждение
            valence, arousal = await self._calculate_valence_arousal(features)
            
            # Создаем объект анализа эмоций
            emotion_analysis = EmotionAnalysis(
                primary_emotion=emotion_result['primary_emotion'],
                confidence=emotion_result['confidence'],
                emotions_distribution=emotion_result['distribution'],
                valence=valence,
                arousal=arousal
            )
            
            self.logger.debug(f"Эмоция распознана: {emotion_analysis.primary_emotion} (уверенность: {emotion_analysis.confidence:.2f})")
            return emotion_analysis
            
        except Exception as e:
            self.logger.error(f"Ошибка распознавания эмоций: {e}")
            return None
    
    async def _extract_acoustic_features(self, audio: np.ndarray) -> Dict[str, float]:
        """
        Извлечение акустических признаков для анализа эмоций
        
        Args:
            audio: Аудио данные
            
        Returns:
            Словарь с акустическими признаками
        """
        try:
            features = {}
            
            # 1. MFCC (Mel-frequency cepstral coefficients)
            mfccs = librosa.feature.mfcc(
                y=audio, 
                sr=self.sample_rate, 
                n_mfcc=13
            )
            features['mfcc_mean'] = float(np.mean(mfccs))
            features['mfcc_std'] = float(np.std(mfccs))
            
            # 2. Спектральные признаки
            spectral_centroids = librosa.feature.spectral_centroid(
                y=audio, 
                sr=self.sample_rate
            )[0]
            features['spectral_centroid_mean'] = float(np.mean(spectral_centroids))
            features['spectral_centroid_std'] = float(np.std(spectral_centroids))
            
            # 3. Спектральная полоса пропускания
            spectral_bandwidth = librosa.feature.spectral_bandwidth(
                y=audio, 
                sr=self.sample_rate
            )[0]
            features['spectral_bandwidth_mean'] = float(np.mean(spectral_bandwidth))
            
            # 4. Спектральная плостность
            spectral_rolloff = librosa.feature.spectral_rolloff(
                y=audio, 
                sr=self.sample_rate
            )[0]
            features['spectral_rolloff_mean'] = float(np.mean(spectral_rolloff))
            
            # 5. Zero crossing rate
            zero_crossing_rate = librosa.feature.zero_crossing_rate(audio)[0]
            features['zero_crossing_rate_mean'] = float(np.mean(zero_crossing_rate))
            
            # 6. RMS энергия
            rms = librosa.feature.rms(y=audio)[0]
            features['rms_mean'] = float(np.mean(rms))
            features['rms_std'] = float(np.std(rms))
            
            # 7. Pitch (высота тона)
            pitches, magnitudes = librosa.piptrack(
                y=audio, 
                sr=self.sample_rate,
                fmin=librosa.note_to_hz('C2'),
                fmax=librosa.note_to_hz('C7')
            )
            
            # Получаем доминирующие частоты
            dominant_pitches = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    dominant_pitches.append(pitch)
            
            if dominant_pitches:
                features['pitch_mean'] = float(np.mean(dominant_pitches))
                features['pitch_std'] = float(np.std(dominant_pitches))
            else:
                features['pitch_mean'] = 0.0
                features['pitch_std'] = 0.0
            
            # 8. Темп речи (аппроксимация)
            energy = []
            frame_length = int(0.025 * self.sample_rate)
            hop_length = int(0.010 * self.sample_rate)
            
            for i in range(0, len(audio) - frame_length, hop_length):
                frame = audio[i:i + frame_length]
                frame_energy = np.sum(frame ** 2)
                energy.append(frame_energy)
            
            if energy:
                energy = np.array(energy)
                features['speaking_rate'] = float(len(energy) / (len(audio) / self.sample_rate))
            else:
                features['speaking_rate'] = 0.0
            
            return features
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения акустических признаков: {e}")
            return {}
    
    async def _analyze_emotions_from_features(
        self, 
        features: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Анализ эмоций на основе акустических признаков
        
        Args:
            features: Словарь с акустическими признаками
            
        Returns:
            Результат анализа эмоций
        """
        try:
            # Простая эвристическая модель на основе акустических признаков
            # В реальной реализации здесь должна быть обученная модель
            
            # Инициализируем распределение эмоций
            emotion_scores = {emotion: 0.0 for emotion in self.emotions}
            
            # Анализируем признаки и присваиваем баллы эмоциям
            
            # 1. Анализ pitch (высота тона)
            pitch_mean = features.get('pitch_mean', 0.0)
            pitch_std = features.get('pitch_std', 0.0)
            
            if pitch_mean > 200:  # Высокий pitch
                emotion_scores['happy'] += 0.3
                emotion_scores['surprised'] += 0.2
            elif pitch_mean < 100:  # Низкий pitch
                emotion_scores['sad'] += 0.3
                emotion_scores['angry'] += 0.2
            
            if pitch_std > 50:  # Нестабильный pitch
                emotion_scores['fearful'] += 0.2
                emotion_scores['angry'] += 0.1
            
            # 2. Анализ энергии (RMS)
            rms_mean = features.get('rms_mean', 0.0)
            rms_std = features.get('rms_std', 0.0)
            
            if rms_mean > 0.1:  # Высокая энергия
                emotion_scores['happy'] += 0.2
                emotion_scores['angry'] += 0.2
            elif rms_mean < 0.02:  # Низкая энергия
                emotion_scores['sad'] += 0.3
                emotion_scores['fearful'] += 0.2
            
            if rms_std > 0.05:  # Нестабильная энергия
                emotion_scores['angry'] += 0.2
                emotion_scores['fearful'] += 0.1
            
            # 3. Анализ спектральных характеристик
            spectral_centroid = features.get('spectral_centroid_mean', 0.0)
            
            if spectral_centroid > 2000:  # Яркий звук
                emotion_scores['happy'] += 0.2
                emotion_scores['surprised'] += 0.1
            elif spectral_centroid < 1000:  # Тусклый звук
                emotion_scores['sad'] += 0.2
                emotion_scores['disgusted'] += 0.1
            
            # 4. Анализ темпа речи
            speaking_rate = features.get('speaking_rate', 0.0)
            
            if speaking_rate > 6:  # Быстрая речь
                emotion_scores['happy'] += 0.2
                emotion_scores['angry'] += 0.2
            elif speaking_rate < 3:  # Медленная речь
                emotion_scores['sad'] += 0.2
                emotion_scores['fearful'] += 0.1
            
            # 5. Zero crossing rate (показатель шумности)
            zcr = features.get('zero_crossing_rate_mean', 0.0)
            
            if zcr > 0.1:  # Высокая шумность
                emotion_scores['angry'] += 0.2
                emotion_scores['fearful'] += 0.1
            
            # Нормализуем баллы
            total_score = sum(emotion_scores.values())
            if total_score > 0:
                emotion_scores = {k: v / total_score for k, v in emotion_scores.items()}
            
            # Определяем основную эмоцию
            primary_emotion = max(emotion_scores, key=emotion_scores.get)
            confidence = emotion_scores[primary_emotion]
            
            # Если уверенность слишком низкая, считаем нейтральной
            if confidence < 0.3:
                primary_emotion = "neutral"
                emotion_scores["neutral"] = 0.5
                # Перераспределяем остальные эмоции
                for emotion in self.emotions:
                    if emotion != "neutral":
                        emotion_scores[emotion] *= 0.5
            
            return {
                'primary_emotion': primary_emotion,
                'confidence': confidence,
                'distribution': emotion_scores
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка анализа эмоций: {e}")
            return {
                'primary_emotion': 'neutral',
                'confidence': 0.5,
                'distribution': {emotion: 0.0 for emotion in self.emotions}
            }
    
    async def _calculate_valence_arousal(
        self, 
        features: Dict[str, float]
    ) -> Tuple[float, float]:
        """
        Вычисление валентности и возбуждения
        
        Args:
            features: Словарь с акустическими признаками
            
        Returns:
            (valence, arousal) где значения от -1 до 1
        """
        try:
            # Валентность (позитив/негатив) от -1 до 1
            valence = 0.0
            
            # Положительные признаки
            if features.get('pitch_mean', 0.0) > 150:
                valence += 0.2
            if features.get('rms_mean', 0.0) > 0.05:
                valence += 0.1
            if features.get('spectral_centroid_mean', 0.0) > 1500:
                valence += 0.1
            
            # Отрицательные признаки
            if features.get('pitch_mean', 0.0) < 100:
                valence -= 0.2
            if features.get('rms_mean', 0.0) < 0.02:
                valence -= 0.2
            if features.get('spectral_centroid_mean', 0.0) < 1000:
                valence -= 0.1
            
            # Нормализуем к диапазону [-1, 1]
            valence = max(-1.0, min(1.0, valence))
            
            # Возбуждение (активность) от 0 до 1
            arousal = 0.0
            
            # Высокая активность
            if features.get('rms_mean', 0.0) > 0.08:
                arousal += 0.3
            if features.get('speaking_rate', 0.0) > 5:
                arousal += 0.2
            if features.get('pitch_std', 0.0) > 40:
                arousal += 0.2
            
            # Низкая активность
            if features.get('rms_mean', 0.0) < 0.03:
                arousal -= 0.2
            if features.get('speaking_rate', 0.0) < 3:
                arousal -= 0.2
            
            # Нормализуем к диапазону [0, 1]
            arousal = max(0.0, min(1.0, arousal))
            
            return valence, arousal
            
        except Exception as e:
            self.logger.error(f"Ошибка вычисления валентности и возбуждения: {e}")
            return 0.0, 0.5  # Нейтральные значения по умолчанию
    
    async def analyze_batch(
        self, 
        audio_segments: List[AudioSegment]
    ) -> List[EmotionAnalysis]:
        """
        Пакетный анализ эмоций в нескольких аудио сегментах
        
        Args:
            audio_segments: Список аудио сегментов
            
        Returns:
            Список анализов эмоций
        """
        emotions = []
        
        for segment in audio_segments:
            emotion = await self.recognize_emotion(segment)
            if emotion:
                emotions.append(emotion)
        
        return emotions
    
    async def health_check(self) -> bool:
        """Проверка работоспособности сервиса"""
        try:
            # Создаем тестовый аудио сегмент
            test_audio = np.random.randn(1600).astype(np.float32) * 0.1
            test_segment = AudioSegment(
                start_time=0.0,
                end_time=0.1,
                duration=0.1,
                audio_data=test_audio.tobytes(),
                confidence=1.0,
                is_speech=True
            )
            
            # Пытаемся проанализировать
            result = await self.recognize_emotion(test_segment)
            
            return result is not None
            
        except Exception as e:
            self.logger.error(f"Ошибка health check: {e}")
            return False
