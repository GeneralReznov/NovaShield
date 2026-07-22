"""
Uses librosa for real spectral feature analysis.
Designed to be replaced with AASIST model when pretrained weights are available.
"""

import numpy as np
import time
import logging
from typing import Dict

from app.core.config import settings
from app.utils.security import FileValidator

logger = logging.getLogger(__name__)


class VoiceAntiSpoofing:
    """
    Hybrid voice anti-spoofing:
    1. Trained GradientBoosting model (if available) for primary classification
    2. Spectral feature heuristics for explainability
    """

    def __init__(self):
        self.ml_model = None
        self.scaler = None
        self.advanced_voice_model = None # Stub for Wav2Vec 2.0 / HuBERT
        self._advanced_loaded = False
        self._load_model()
        # self._load_advanced_model() is now called lazily inside analyze()
        
    def _load_advanced_model(self):
        """
        Load the Advanced Voice Model (HuggingFace Deepfake Audio Classifier).
        """
        try:
            from transformers import pipeline
            import torch
            
            # This model is fine-tuned on ASVspoof for real/fake classification
            model_name = "MelodyMachine/Deepfake-audio-detection-V2"
            logger.info(f"Loading SOTA Audio model: {model_name}...")
            
            device = 0 if torch.cuda.is_available() else -1
            self.advanced_voice_model = pipeline(
                "audio-classification", 
                model=model_name, 
                device=device
            )
            
            logger.info(f"Successfully loaded Advanced Voice (Wav2Vec2) model from HuggingFace.")
        except Exception as e:
            logger.error(f"Failed to load Advanced Voice model: {e}")
            self.advanced_voice_model = None

    def _load_model(self):
        """Load trained voice model if available and cryptographically valid"""
        import os
        model_name = getattr(settings, 'VOICE_MODEL', 'voice_antispoofing.pkl')
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'pretrained', model_name)
        paths = [
            model_path,
            f'models/pretrained/{model_name}',
            f'../models/pretrained/{model_name}',
            f'../../models/pretrained/{model_name}'
        ]
        for path in paths:
            try:
                if not os.path.exists(path):
                    continue

                # Cryptographic check
                expected_hash = getattr(settings, 'VOICE_MODEL_SHA256', None)
                if not FileValidator.verify_model_hash(path, expected_hash):
                    logger.warning(f"Voice model {path} failed cryptographic validation. Skipping this path.")
                    continue

                import joblib
                data = joblib.load(path)
                if isinstance(data, dict):
                    # Validate model wasn't trained on synthetic random data
                    accuracy = data.get('accuracy', 0)
                    if accuracy > 0.99:
                        logger.warning(f"Voice model at {path} has suspiciously high accuracy ({accuracy:.3f}) — likely trained on synthetic data. Skipping.")
                        continue
                    self.ml_model = data.get('model')
                    self.scaler = data.get('scaler')
                else:
                    self.ml_model = data
                logger.info(f"Loaded voice model from {path}")
                return
            except Exception as e:
                logger.warning(f"Could not load voice model from {path}: {e}")
                continue
        logger.warning("No valid voice model found. Activating Degraded Heuristic-Only Mode.")

    def _extract_ml_features(self, y, sr):
        """Extract 89 spectral features matching the training pipeline.
        Must produce the same feature vector as training/train_voice.py:extract_features()
        """
        try:
            import librosa
            if len(y) < sr:
                y = np.pad(y, (0, sr - len(y)))

            features = []
            # MFCCs (13 coefficients × 4 stats = 52)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            for mfcc in mfccs:
                features.extend([np.mean(mfcc), np.std(mfcc), np.min(mfcc), np.max(mfcc)])

            # Spectral features (5 × 2 stats = 10)
            for feat_fn in [librosa.feature.spectral_centroid, librosa.feature.spectral_bandwidth,
                            librosa.feature.spectral_rolloff, librosa.feature.spectral_flatness,
                            librosa.feature.zero_crossing_rate]:
                f = feat_fn(y=y, sr=sr) if 'sr' in feat_fn.__code__.co_varnames else feat_fn(y=y)
                features.extend([float(np.mean(f)), float(np.std(f))])

            # Chroma (12 × 2 = 24)
            chroma = librosa.feature.chroma_stft(y=y, sr=sr)
            for c in chroma:
                features.extend([np.mean(c), np.std(c)])

            # RMS energy (2)
            rms = librosa.feature.rms(y=y)
            features.extend([float(np.mean(rms)), float(np.std(rms))])

            # Tempo (1)
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(float(tempo) if np.isscalar(tempo) else float(tempo[0]))

            return np.array(features, dtype=np.float32)  # 89 features total
        except Exception as e:
            logger.debug(f"ML feature extraction failed: {e}")
            return None

    def analyze(self, audio_path: str) -> Dict:
        if not self._advanced_loaded:
            self._load_advanced_model()
            self._advanced_loaded = True
            
        start_time = time.time()

        try:
            import librosa

            # Load FULL audio file (Removed 30-second limit)
            y, sr = librosa.load(audio_path, sr=16000)

            if len(y) < sr * 0.5:  # Less than 0.5 seconds
                return {
                    "is_fake": False,
                    "confidence": 0.0,
                    "spoof_type": None,
                    "explanation": "Audio too short for reliable analysis (minimum 0.5 seconds)",
                    "processing_time": round(time.time() - start_time, 2),
                    "recommended_action": "Please provide a longer audio sample"
                }

            scores = {}

            # 1. Spectral Centroid Analysis (synthetic speech tends to have different centroid patterns)
            spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            centroid_mean = float(np.mean(spectral_centroid))
            centroid_std = float(np.std(spectral_centroid))
            centroid_cv = centroid_std / (centroid_mean + 1e-6)

            # Very uniform centroid = potentially synthetic
            if centroid_cv < 0.15:
                scores['spectral_centroid'] = 20
            elif centroid_cv < 0.25:
                scores['spectral_centroid'] = 10
            else:
                scores['spectral_centroid'] = 0

            # 2. Spectral Flatness (synthetic audio often has higher flatness)
            spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
            flatness_mean = float(np.mean(spectral_flatness))

            if flatness_mean > 0.3:
                scores['spectral_flatness'] = 15  # Very flat = noise-like (replay attack)
            elif flatness_mean > 0.15:
                scores['spectral_flatness'] = 8
            else:
                scores['spectral_flatness'] = 0

            # 3. MFCC Analysis (voice conversion changes MFCC patterns)
            mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_variance = float(np.mean(np.var(mfccs, axis=1)))
            mfcc_delta = librosa.feature.delta(mfccs)
            mfcc_delta_var = float(np.mean(np.var(mfcc_delta, axis=1)))

            # Low MFCC variance = potentially monotone/synthetic
            if mfcc_variance < 50:
                scores['mfcc_pattern'] = 15
            elif mfcc_variance < 100:
                scores['mfcc_pattern'] = 8
            else:
                scores['mfcc_pattern'] = 0

            # Low delta variance = lacks natural speech dynamics
            if mfcc_delta_var < 5:
                scores['temporal_dynamics'] = 15
            elif mfcc_delta_var < 15:
                scores['temporal_dynamics'] = 8
            else:
                scores['temporal_dynamics'] = 0

            # 4. Zero-Crossing Rate (TTS has more regular patterns)
            zcr = librosa.feature.zero_crossing_rate(y)[0]
            zcr_mean = float(np.mean(zcr))
            zcr_std = float(np.std(zcr))
            zcr_cv = zcr_std / (zcr_mean + 1e-6)

            if zcr_cv < 0.3:
                scores['zero_crossing'] = 10
            else:
                scores['zero_crossing'] = 0

            # 5. Spectral Bandwidth (synthetic audio often has narrower bandwidth)
            spectral_bw = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
            bw_mean = float(np.mean(spectral_bw))

            if bw_mean < 1500:
                scores['bandwidth'] = 10  # Narrow bandwidth suspicious
            elif bw_mean < 2000:
                scores['bandwidth'] = 5
            else:
                scores['bandwidth'] = 0

            # 6. Spectral Rolloff
            rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
            rolloff_mean = float(np.mean(rolloff))

            if rolloff_mean < 3000:
                scores['rolloff'] = 10
            else:
                scores['rolloff'] = 0

            # 7. RMS Energy Dynamics
            rms = librosa.feature.rms(y=y)[0]
            rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-6))

            if rms_cv < 0.3:
                scores['energy_dynamics'] = 5  # Too uniform energy
            else:
                scores['energy_dynamics'] = 0

            # Total scoring
            total_score = sum(scores.values())

            # === ML MODEL PREDICTION (Chunked) ===
            ml_prediction = None
            ml_confidence = None
            detection_method = "Spectral Heuristic"

            # Split audio into 30-second chunks (30 * sr)
            chunk_size = 30 * sr
            chunks = [y[i:i + chunk_size] for i in range(0, len(y), chunk_size)]
            
            highest_confidence = 0.0
            is_fake_any_chunk = False

            if self.advanced_voice_model is not None:
                try:
                    for chunk in chunks:
                        if len(chunk) < sr: continue
                        results = self.advanced_voice_model(chunk)
                        best_match = max(results, key=lambda x: x['score'])
                        label = best_match['label'].lower()
                        
                        chunk_is_fake = 'fake' in label or 'spoof' in label or label == 'label_1' or 'synthetic' in label or 'ai' in label
                        chunk_conf = float(best_match['score'] * 100)
                        
                        if chunk_is_fake and chunk_conf > highest_confidence:
                            highest_confidence = chunk_conf
                            is_fake_any_chunk = True
                        elif not chunk_is_fake and not is_fake_any_chunk and chunk_conf > highest_confidence:
                            # Keep highest real confidence if we haven't found a fake yet
                            highest_confidence = chunk_conf

                    ml_prediction = is_fake_any_chunk
                    ml_confidence = highest_confidence
                    detection_method = "HuggingFace Deepfake Audio Classifier (Chunked)"
                    logger.info(f"Advanced Voice ML prediction (Full Audio): IsFake: {ml_prediction}, confidence: {ml_confidence:.1f}%")
                except Exception as e:
                    logger.warning(f"Advanced Voice ML prediction failed, falling back to heuristics: {e}")

            elif self.ml_model is not None:
                try:
                    for chunk in chunks:
                        if len(chunk) < sr: continue
                        ml_features = self._extract_ml_features(chunk, sr)
                        if ml_features is not None:
                            features_array = ml_features.reshape(1, -1)
                            if self.scaler is not None:
                                features_array = self.scaler.transform(features_array)
                            chunk_pred = int(self.ml_model.predict(features_array)[0])
                            chunk_proba = self.ml_model.predict_proba(features_array)[0]
                            chunk_conf = float(max(chunk_proba) * 100)
                            
                            if chunk_pred == 1 and chunk_conf > highest_confidence:
                                highest_confidence = chunk_conf
                                is_fake_any_chunk = True
                            elif chunk_pred == 0 and not is_fake_any_chunk and chunk_conf > highest_confidence:
                                highest_confidence = chunk_conf

                    ml_prediction = is_fake_any_chunk
                    ml_confidence = highest_confidence
                    detection_method = "GradientBoosting ML (Chunked) + Heuristic"
                    logger.info(f"Voice ML prediction (Full Audio): {ml_prediction}, confidence: {ml_confidence:.1f}%")
                except Exception as e:
                    logger.warning(f"Voice ML prediction failed, using heuristics: {e}")

            # === FINAL VERDICT ===
            if ml_prediction is not None:
                is_fake = bool(ml_prediction)
                confidence = ml_confidence
            else:
                is_fake = total_score > 35
                confidence = min(98, max(2, total_score))

            # Determine spoof type based on dominant indicators
            spoof_type = None
            if is_fake:
                if scores.get('spectral_flatness', 0) >= 12:
                    spoof_type = "Replay Attack"
                elif scores.get('temporal_dynamics', 0) >= 12 and scores.get('mfcc_pattern', 0) >= 12:
                    spoof_type = "Text-to-Speech (TTS)"
                elif scores.get('mfcc_pattern', 0) >= 12:
                    spoof_type = "Voice Conversion"
                else:
                    spoof_type = "Synthetic Audio"

            processing_time = round(time.time() - start_time, 2)

            if is_fake:
                top_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
                factor_names = {
                    'spectral_centroid': 'unnatural spectral centroid patterns',
                    'spectral_flatness': 'high spectral flatness (noise-like)',
                    'mfcc_pattern': 'anomalous MFCC patterns',
                    'temporal_dynamics': 'lack of natural speech dynamics',
                    'zero_crossing': 'overly regular zero-crossing rate',
                    'bandwidth': 'narrow spectral bandwidth',
                    'rolloff': 'low spectral rolloff',
                    'energy_dynamics': 'uniform energy dynamics'
                }
                factors_text = ", ".join([factor_names.get(f[0], f[0]) for f in top_factors if f[1] > 3])
                ml_note = f" [{detection_method}]" if ml_confidence is not None else ""
                explanation = f"{ml_note} Potential voice spoofing detected ({spoof_type}). Indicators: {factors_text}."
            else:
                explanation = f"[{detection_method}] Spectral analysis found natural voice characteristics. Audio appears authentic."

            return {
                "is_fake": bool(is_fake),
                "confidence": float(round(confidence, 1)),
                "detection_method": detection_method,
                "spoof_type": spoof_type,
                "spectral_centroid_mean": round(centroid_mean, 2),
                "spectral_flatness": round(flatness_mean, 4),
                "mfcc_variance": round(mfcc_variance, 2),
                "zero_crossing_rate": round(zcr_mean, 4),
                "explanation": explanation,
                "processing_time": processing_time,
                "recommended_action": "Reject voice authentication — spoofing indicators detected" if is_fake else "Voice appears authentic based on spectral analysis",
                "analysis_details": scores
            }

        except ImportError:
            logger.warning("librosa not available, using basic analysis")
            return self._basic_analyze(audio_path, start_time)
        except Exception as e:
            logger.error(f"Voice analysis error: {e}")
            return {
                "is_fake": False,
                "confidence": 0.0,
                "spoof_type": None,
                "explanation": f"Analysis encountered an error: {str(e)}",
                "processing_time": round(time.time() - start_time, 2),
                "recommended_action": "Please try again with a different audio file"
            }

    def _basic_analyze(self, audio_path: str, start_time: float) -> Dict:
        """Fallback analysis when librosa is not available"""
        import wave

        try:
            with wave.open(audio_path, 'r') as wf:
                n_channels = wf.getnchannels()
                sample_width = wf.getsampwidth()
                framerate = wf.getframerate()
                n_frames = wf.getnframes()
                duration = n_frames / framerate

                frames = wf.readframes(min(n_frames, framerate * 10))
                audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32)

                if len(audio_data) == 0:
                    return {
                        "is_fake": False, "confidence": 0.0, "spoof_type": None,
                        "explanation": "Empty audio file",
                        "processing_time": round(time.time() - start_time, 2),
                        "recommended_action": "Upload a valid audio file"
                    }

                rms = np.sqrt(np.mean(audio_data ** 2))
                zcr = np.sum(np.diff(np.sign(audio_data)) != 0) / len(audio_data)

                score = 0
                if zcr < 0.05 or zcr > 0.5:
                    score += 20
                if rms < 100:
                    score += 15

                is_fake = score > 25
                return {
                    "is_fake": bool(is_fake),
                    "confidence": float(min(score, 60)),
                    "spoof_type": "Unknown" if is_fake else None,
                    "explanation": "Basic audio analysis (librosa unavailable)" + (
                        " — anomalous characteristics detected" if is_fake else " — no anomalies found"),
                    "processing_time": round(time.time() - start_time, 2),
                    "recommended_action": "Install librosa for comprehensive analysis"
                }
        except Exception:
            return {
                "is_fake": False, "confidence": 0.0, "spoof_type": None,
                "explanation": "Could not parse audio file (unsupported format for basic analysis)",
                "processing_time": round(time.time() - start_time, 2),
                "recommended_action": "Please upload a WAV file for analysis"
            }


voice_detector = VoiceAntiSpoofing()
