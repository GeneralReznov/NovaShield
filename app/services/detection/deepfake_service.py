"""
REAL AI: TSMEfficientNet (Temporal Shift Module for 3D Video Awareness) + OpenCV heuristics
"""

import cv2
import numpy as np
import os
import time
import logging
from typing import Dict

from app.core.config import settings
from app.utils.security import FileValidator

logger = logging.getLogger(__name__)


class DeepfakeDetector:
    def __init__(self):
        try:
            from facenet_pytorch import MTCNN
            import torch
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.face_detector = MTCNN(keep_all=True, device=device)
        except ImportError:
            self.face_detector = None
            logger.warning("facenet-pytorch not installed. Deepfake detection requires it.")
            
            # Use OpenCV DNN (ResNet SSD) as the enterprise-grade fallback
            prototxt_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'pretrained', 'deploy.prototxt')
            caffemodel_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'pretrained', 'res10_300x300_ssd_iter_140000.caffemodel')
            if os.path.exists(prototxt_path) and os.path.exists(caffemodel_path):
                self.dnn_face_detector = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)
            else:
                self.dnn_face_detector = None
                logger.error("No valid DNN or MTCNN face detector found. Hardware requires OpenCV DNN at minimum.")
                
        try:
            self.eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_eye.xml'
            )
            # Face cascade for live frame fallback
            self.face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        except AttributeError:
            self.eye_cascade = None
            self.face_cascade = None
            logger.warning("cv2.CascadeClassifier not available in this OpenCV build. Haar cascades disabled.")
        except Exception as e:
            self.eye_cascade = None
            self.face_cascade = None
            logger.warning(f"Failed to load OpenCV cascades: {e}")
        
        # Load trained EfficientNet model
        self.ml_model = None
        self.transform = None
        self.advanced_model = None # Stub for Spatio-Temporal Model
        self.num_segments = 8 # TSM segment size
        
        self._load_model()
        self._load_advanced_model()
        
    def _load_advanced_model(self):
        """
        Load Vision Transformer (ViT) for SOTA spatial deepfake detection.
        """
        try:
            from transformers import AutoImageProcessor, AutoModelForImageClassification
            import torch
            
            # Using a public fine-tuned ViT for deepfake detection
            model_name = "dima806/deepfake_vs_real_image_detection"
            logger.info(f"Loading SOTA ViT model: {model_name}...")
            
            self.vit_processor = AutoImageProcessor.from_pretrained(model_name)
            self.vit_model = AutoModelForImageClassification.from_pretrained(model_name)
            
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.vit_model.to(self.device)
            self.vit_model.eval()
            self.advanced_model = True
            
            # Determine which class index corresponds to FAKE/deepfake
            self.vit_fake_label_idx = 0  # default fallback
            id2label = getattr(self.vit_model.config, 'id2label', {})
            for idx, label in id2label.items():
                label_lower = str(label).lower()
                if any(word in label_lower for word in ['fake', 'deepfake', 'manipulated', 'synthetic', 'ai']):
                    self.vit_fake_label_idx = int(idx)
                    break
            
            logger.info("Successfully loaded ViT Deepfake model.")
        except Exception as e:
            logger.error(f"Failed to load ViT Advanced model: {e}")
            self.advanced_model = None
            self.vit_model = None
            self.vit_fake_label_idx = 0

    def _load_model(self):
        """Load the user's previously trained static EfficientNet model (model_best.pth)"""
        import os
        import torch
        import torchvision.transforms as transforms
        from torchvision import models
        import torch.nn as nn
        
        # Use the user's existing model
        model_name = 'model_best.pth'
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'pretrained', model_name)
        model_paths = [
            model_path,
            f'models/pretrained/{model_name}',
            f'../models/pretrained/{model_name}',
            f'../../models/pretrained/{model_name}'
        ]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for path in model_paths:
            try:
                if not os.path.exists(path):
                    continue

                # Load standard EfficientNet-B0
                model = models.efficientnet_b0(weights=None)
                in_features = model.classifier[1].in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.4, inplace=True),
                    nn.Linear(in_features, 1)
                )
                
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.ml_model = model
                self._model_type = 'logits'

                # Image transform
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                logger.info(f"Loaded previously trained deepfake model from {path}")
                return
            except Exception as e:
                logger.warning(f"Could not load deepfake model from {path}: {e}")
                continue
                
        self._model_type = None
        logger.warning("No valid deepfake model found. Activating Degraded Heuristic-Only Mode.")
        """Load trained TSMEfficientNet deepfake model"""
        import os
        import torch
        import torchvision.transforms as transforms
        from app.services.detection.tsm_model import TSMEfficientNet
        
        model_name = getattr(settings, 'DEEPFAKE_MODEL', 'deepfake_tsm_efficientnet_b0.pth')
        model_path = os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', 'models', 'pretrained', model_name)
        model_paths = [
            model_path,
            f'models/pretrained/{model_name}',
            f'../models/pretrained/{model_name}',
            f'../../models/pretrained/{model_name}'
        ]
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        for path in model_paths:
            try:
                if not os.path.exists(path):
                    continue

                # Load TSM Model
                model = TSMEfficientNet(num_classes=1, num_segments=self.num_segments, pretrained=False)
                checkpoint = torch.load(path, map_location='cpu', weights_only=False)
                state_dict = checkpoint.get('state_dict', checkpoint.get('model_state_dict', checkpoint))
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.ml_model = model
                self._model_type = 'logits'

                # Image transform
                self.transform = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

                logger.info(f"Loaded TSMEfficientNet deepfake model from {path}")
                return
            except Exception as e:
                logger.warning(f"Could not load deepfake model from {path}: {e}")
                continue
                
        self._model_type = None
        logger.warning("No valid deepfake model found. Activating Degraded Heuristic-Only Mode.")

    def _predict_sequence(self, face_sequence):
        """Run ML predictions on a sequence of face crops for Static EfficientNet."""
        probs = {"efficientnet": None, "vit": None}
        
        try:
            import torch
            
            # Predict using static EfficientNet (average over the sequence)
            if self.ml_model is not None and self.transform is not None and len(face_sequence) > 0:
                tensor_frames = []
                for face_roi in face_sequence:
                    face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
                    tensor_frames.append(self.transform(face_rgb))
                    
                input_tensor = torch.stack(tensor_frames).to(self.device) # (T, C, H, W)
                
                with torch.no_grad():
                    output = self.ml_model(input_tensor) # output is (T, 1)
                    prob_frames = torch.sigmoid(output).squeeze()
                    if prob_frames.dim() == 0:
                        prob = prob_frames.item()
                    else:
                        prob = torch.mean(prob_frames).item() # Average probability across frames
                    probs["efficientnet"] = prob
                        
            # Vision Transformer (ViT) Prediction on the middle frame
            if getattr(self, 'advanced_model', None) and getattr(self, 'vit_model', None) and len(face_sequence) > 0:
                try:
                    mid_frame = cv2.cvtColor(face_sequence[len(face_sequence)//2], cv2.COLOR_BGR2RGB)
                    inputs = self.vit_processor(images=mid_frame, return_tensors="pt").to(self.device)
                    with torch.no_grad():
                        vit_outputs = self.vit_model(**inputs)
                        logits = vit_outputs.logits
                        vit_probs = torch.softmax(logits, dim=1)
                        fake_idx = getattr(self, 'vit_fake_label_idx', 0)
                        probs["vit"] = vit_probs[0][fake_idx].item()
                except Exception as e:
                    logger.error(f"ViT prediction failed: {e}")
                    
            return probs
        except Exception as e:
            logger.error(f"ML prediction failed: {e}")
            return probs

    def analyze(self, video_path: str, debug_mode: bool = True) -> dict:
        """
        Analyze a video file for deepfake artifacts using multimodal AI fusion.
        Returns a detailed report.
        """
        start_time = time.time()
        logger.info(f"Starting deepfake analysis: {video_path}")
        
        # --- DEBUG MODE INITIALIZATION ---
        debug_dir = None
        if debug_mode:
            import uuid
            debug_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))), "debug_crops", str(uuid.uuid4())[:8])
            os.makedirs(debug_dir, exist_ok=True)
            logger.info(f"[DEBUG] Saving face crops to: {debug_dir}")

        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        file_size = os.path.getsize(video_path)
        if file_size < 100:
            raise ValueError(f"Video file too small ({file_size} bytes) — likely corrupt upload")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            file_ext = os.path.splitext(video_path)[1].lower()
            ext_hint = ""
            if file_ext in ('.mkv', '.avi'):
                ext_hint = f" Note: {file_ext.upper()} files require codec support. Try converting to MP4 (H.264)."
            return {
                "is_fake": False, "confidence": 0.0,
                "detection_method": "Heuristic Only",
                "frames_analyzed": 0, "faces_detected": 0,
                "explanation": f"Could not open video file — the format may not be supported.{ext_hint}",
                "processing_time": round(time.time() - start_time, 2),
                "recommended_action": "Please re-upload as MP4 (H.264) format"
            }

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        if width > 3840 or height > 2160:
            cap.release()
            return {
                "is_fake": False, "confidence": 0.0,
                "detection_method": "Heuristic Only",
                "frames_analyzed": 0, "faces_detected": 0,
                "explanation": f"Video resolution too high ({width}x{height}). Max allowed is 4K (3840x2160).",
                "processing_time": round(time.time() - start_time, 2),
                "recommended_action": "Please upload a lower resolution video"
            }
            
        min_expected_size = total_frames * 500
        if total_frames > 0 and file_size < min_expected_size and total_frames > 1000:
            cap.release()
            return {
                "is_fake": False, "confidence": 0.0,
                "detection_method": "Heuristic Only",
                "frames_analyzed": 0, "faces_detected": 0,
                "explanation": "Invalid video file metadata. Potential decompression bomb detected.",
                "processing_time": round(time.time() - start_time, 2),
                "recommended_action": "Please upload a valid video file"
            }

        target_frames = 150
        sample_interval = max(1, int(total_frames / target_frames)) if total_frames > 0 else max(1, int(fps / 5))

        frames_analyzed = 0
        faces_detected = 0
        face_sizes = []
        face_positions = []
        edge_scores = []
        blink_count = 0
        color_histograms = []
        compression_scores = []
        ml_predictions = [] 
        fft_scores = []      
        mouth_scores = []    
        rppg_signals = []    
        
        face_sequence_buffer = []

        frame_idx = 0
        while cap.isOpened() and frames_analyzed < target_frames:
            if time.time() - start_time > settings.INFERENCE_TIMEOUT:
                break

            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                frames_analyzed += 1
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                faces = []
                if hasattr(self, 'face_detector') and self.face_detector is not None:
                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    boxes, _ = self.face_detector.detect(rgb_frame)
                    if boxes is not None:
                        for box in boxes:
                            x1, y1, x2, y2 = [int(b) for b in box]
                            w, h = x2 - x1, y2 - y1
                            if w > 30 and h > 30:
                                faces.append((x1, y1, w, h))
                else:
                    if hasattr(self, 'dnn_face_detector') and self.dnn_face_detector is not None:
                        h_f, w_f = frame.shape[:2]
                        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
                        self.dnn_face_detector.setInput(blob)
                        detections = self.dnn_face_detector.forward()
                        
                        for i in range(detections.shape[2]):
                            confidence = detections[0, 0, i, 2]
                            if confidence > 0.5: 
                                box = detections[0, 0, i, 3:7] * np.array([w_f, h_f, w_f, h_f])
                                (startX, startY, endX, endY) = box.astype("int")
                                fw, fh = endX - startX, endY - startY
                                if fw > 30 and fh > 30:
                                    faces.append((startX, startY, fw, fh))
                    else:
                        faces = []

                if len(faces) > 0:
                    faces_detected += 1
                    x, y, w, h = faces[0]
                    
                    pad_w = 0
                    pad_h = 0
                    
                    new_x = max(0, x - pad_w // 2)
                    new_y = max(0, y - pad_h // 2)
                    new_w = min(frame.shape[1] - new_x, w + pad_w)
                    new_h = min(frame.shape[0] - new_y, h + pad_h)
                    
                    x, y, w, h = new_x, new_y, new_w, new_h
                    
                    face_sizes.append((w, h))
                    face_positions.append((x + w // 2, y + h // 2))

                    face_roi_color = frame[y:y+h, x:x+w]
                    if face_roi_color.size > 0:
                        green_channel = face_roi_color[:, :, 1]
                        rppg_signals.append(np.mean(green_channel))
                        
                        if w > 50 and h > 50:
                            face_sequence_buffer.append(face_roi_color)
                            
                            # TSM Sequence Processing
                            if len(face_sequence_buffer) == self.num_segments:
                                ml_pred = self._predict_sequence(face_sequence_buffer)
                                if ml_pred.get("efficientnet") is not None or ml_pred.get("vit") is not None:
                                    ml_predictions.append(ml_pred)
                                face_sequence_buffer.clear()

                    face_roi = gray[y:y+h, x:x+w]
                    if face_roi.size > 0:
                        laplacian = cv2.Laplacian(face_roi, cv2.CV_64F)
                        edge_scores.append(laplacian.var())

                        f_transform = np.fft.fft2(face_roi)
                        f_shift = np.fft.fftshift(f_transform)
                        magnitude_spectrum = 20 * np.log(np.abs(f_shift) + 1e-8)
                        fh_f, fw_f = magnitude_spectrum.shape
                        cy, cx = fh_f // 2, fw_f // 2
                        y_grid, x_grid = np.ogrid[:fh_f, :fw_f]
                        mask = (x_grid - cx)**2 + (y_grid - cy)**2 > (min(cx, cy) * 0.8)**2
                        high_freq_mag = np.mean(magnitude_spectrum[mask]) if np.any(mask) else 0
                        fft_scores.append(high_freq_mag)

                    mouth_roi = gray[y + int(h * 0.66):y + h, x + int(w * 0.2):x + int(w * 0.8)]
                    if mouth_roi.size > 0:
                        mouth_laplacian = cv2.Laplacian(mouth_roi, cv2.CV_64F)
                        mouth_scores.append(mouth_laplacian.var())

                    eyes = []
                    if self.eye_cascade:
                        eyes = self.eye_cascade.detectMultiScale(
                            face_roi, scaleFactor=1.1, minNeighbors=3
                        )
                    if len(eyes) < 2:
                        blink_count += 1

                    face_color = frame[y:y+h, x:x+w]
                    if face_color.size > 0:
                        hist = cv2.calcHist([face_color], [0, 1, 2], None,
                                           [8, 8, 8], [0, 256, 0, 256, 0, 256])
                        hist = cv2.normalize(hist, hist).flatten()
                        color_histograms.append(hist)

                if gray.shape[0] >= 8 and gray.shape[1] >= 8:
                    block = gray[:8, :8].astype(np.float32)
                    dct_block = cv2.dct(block)
                    compression_scores.append(np.abs(dct_block).mean())

            frame_idx += 1
            
        # Process remaining buffer by duplicating last frame
        if len(face_sequence_buffer) > 0 and self.ml_model is not None:
            while len(face_sequence_buffer) < self.num_segments:
                face_sequence_buffer.append(face_sequence_buffer[-1])
            ml_pred = self._predict_sequence(face_sequence_buffer)
            if ml_pred.get("efficientnet") is not None or ml_pred.get("vit") is not None:
                ml_predictions.append(ml_pred)
            face_sequence_buffer.clear()

        cap.release()

        ml_confidence = None
        ml_is_fake = None
        vit_prob = None
        composite_score = None
        efficientnet_prob = None
        detection_method = "Heuristic Only"

        if len(ml_predictions) > 0:
            eff_probs = [p["efficientnet"] for p in ml_predictions if p.get("efficientnet") is not None]
            vit_probs_list = [p["vit"] for p in ml_predictions if p.get("vit") is not None]
            
            suspicious_frame_ratio = 0
            if eff_probs:
                efficientnet_prob = float(np.max(eff_probs)) 
                efficientnet_median = float(np.median(eff_probs))
                suspicious_frame_ratio = sum(1 for p in eff_probs if p > 0.60) / len(eff_probs)
            if vit_probs_list:
                vit_prob = float(np.max(vit_probs_list)) 

            if vit_prob is not None and efficientnet_prob is not None:
                composite_score = (
                    efficientnet_median * 0.90 + 
                    suspicious_frame_ratio * 0.10
                )
                ml_is_fake = composite_score > 0.50
                ml_confidence = float(composite_score * 100) if ml_is_fake else float((1 - composite_score) * 100)
                detection_method = "TSMEfficientNet Fusion"
            elif efficientnet_prob is not None:
                composite_score = (
                    efficientnet_median * 0.70 +        
                    efficientnet_prob * 0.20 +          
                    suspicious_frame_ratio * 0.10       
                )
                ml_is_fake = composite_score > 0.50
                ml_confidence = float(composite_score * 100) if ml_is_fake else float((1 - composite_score) * 100)
                detection_method = "TSMEfficientNet + Heuristic"

        scores = {}
        if len(face_sizes) >= 3:
            size_widths = [s[0] for s in face_sizes]
            size_cv = np.std(size_widths) / (np.mean(size_widths) + 1e-6)
            pos_x = [p[0] for p in face_positions]
            pos_jitter = np.std(pos_x) / (np.mean(pos_x) + 1e-6)
            scores['face_consistency'] = round(min(15, (size_cv * 30 + pos_jitter * 30)), 2)
        else: scores['face_consistency'] = 0

        if len(edge_scores) >= 3:
            edge_ratio = np.std(edge_scores) / (np.mean(edge_scores) + 1e-6)
            scores['edge_artifact'] = round(min(25, edge_ratio * 40), 2) if edge_ratio > 0.25 else 0
        else: scores['edge_artifact'] = 0

        if frames_analyzed > 120 and faces_detected > 30:
            blink_ratio = blink_count / faces_detected
            if blink_ratio < 0.02: scores['blink_analysis'] = 15
            elif blink_ratio < 0.08: scores['blink_analysis'] = 10
            elif blink_ratio < 0.15: scores['blink_analysis'] = 5
            else: scores['blink_analysis'] = 0
        else: scores['blink_analysis'] = 0

        if len(color_histograms) >= 3:
            correlations = [cv2.compareHist(color_histograms[i-1], color_histograms[i], cv2.HISTCMP_CORREL) for i in range(1, len(color_histograms))]
            avg_corr = np.mean(correlations)
            scores['color_consistency'] = round(min(15, (1 - avg_corr) * 30), 2) if avg_corr < 0.85 else 0
        else: scores['color_consistency'] = 0

        if len(compression_scores) >= 3:
            comp_cv = np.std(compression_scores) / (np.mean(compression_scores) + 1e-6)
            # Reduced sensitivity: social media/phone compression should not trigger false positives
            # Only flag truly abnormal patterns (threshold raised from 0.3 to 0.6)
            scores['compression_anomaly'] = round(min(10, comp_cv * 15), 2) if comp_cv > 0.6 else 0
        else: scores['compression_anomaly'] = 0

        if len(fft_scores) >= 3:
            fft_cv = np.std(fft_scores) / (np.mean(fft_scores) + 1e-6)
            scores['fft_artifact'] = round(min(12, fft_cv * 40), 2)
        else: scores['fft_artifact'] = 0

        if len(mouth_scores) >= 3:
            mouth_cv = np.std(mouth_scores) / (np.mean(mouth_scores) + 1e-6)
            if mouth_cv < 0.02 or mouth_cv > 1.8: scores['mouth_anomaly'] = 15
            else: scores['mouth_anomaly'] = 0
        else: scores['mouth_anomaly'] = 0

        # Require at least 300 frames for reliable rPPG (short/compressed phone videos give false signals)
        if len(rppg_signals) >= 300:
            signal = np.array(rppg_signals)
            signal_detrended = signal - np.mean(signal)
            freqs = np.fft.rfftfreq(len(signal_detrended), d=1.0/fps)
            fft_power = np.abs(np.fft.rfft(signal_detrended))
            valid_hr_mask = (freqs >= 0.8) & (freqs <= 3.0)
            if np.any(valid_hr_mask):
                snr = np.max(fft_power[valid_hr_mask]) / (np.sum(fft_power) + 1e-6)
                if snr < 0.05: scores['rppg_anomaly'] = 15
                elif snr < 0.10: scores['rppg_anomaly'] = 10
                elif snr < 0.15: scores['rppg_anomaly'] = 5
                else: scores['rppg_anomaly'] = round(max(0, (0.2 - snr) * 40), 2)
            else: scores['rppg_anomaly'] = 15
        else: scores['rppg_anomaly'] = 0

        audio_verdict, audio_confidence = None, 0.0
        try:
            from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio
            from app.services.detection.voice_service import VoiceAntiSpoofing
            import tempfile
            temp_audio = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            temp_audio.close()
            try:
                ffmpeg_extract_audio(video_path, temp_audio.name)
                if os.path.exists(temp_audio.name) and os.path.getsize(temp_audio.name) > 100:
                    voice_detector = VoiceAntiSpoofing()
                    audio_res = voice_detector.analyze(temp_audio.name)
                    if audio_res["success"]:
                        audio_verdict = audio_res["is_fake"]
                        audio_confidence = audio_res["confidence"]
            finally:
                if os.path.exists(temp_audio.name): os.unlink(temp_audio.name)
        except: pass

        heuristic_score = sum(scores.values())

        if ml_is_fake is not None or vit_prob is not None:
            eff_is_fake = ml_is_fake if ml_is_fake is not None else False
            eff_conf = ml_confidence if ml_confidence is not None else 0.0
            
            v_is_fake = vit_prob > 0.50 if vit_prob is not None else False
            v_conf = float(vit_prob * 100) if v_is_fake and vit_prob else 0.0
            
            ml_prob = composite_score if composite_score is not None else (efficientnet_prob if efficientnet_prob is not None else (vit_prob if vit_prob is not None else 0.0))
            heuristic_normalized = min(1.0, heuristic_score / 100.0) 
            
            fusion_score = ml_prob * 0.80 + heuristic_normalized * 0.20
            
            if audio_verdict is not None:
                audio_signal = 1.0 if audio_verdict else 0.0
                fusion_score = fusion_score * 0.85 + audio_signal * 0.15
            
            strong_heuristic_signals = sum(1 for v in scores.values() if v > 3)
            # Raised thresholds to reduce false positives on phone/social media videos
            # Previously 0.40/0.50 — now 0.55/0.65 for higher confidence requirement
            threshold = 0.55 if strong_heuristic_signals >= 3 else 0.65
            
            is_fake = fusion_score > threshold
            if is_fake: confidence = min(99.0, max(55.0, fusion_score * 100 + strong_heuristic_signals * 2))
            else: confidence = min(99.0, max(50.0, (1 - fusion_score) * 100))
                
            methods = []
            if ml_is_fake is not None: methods.append("TSMEfficientNet")
            if v_conf > 0: methods.append("ViT")
            if audio_verdict is not None: methods.append("WavLM-Audio")
            methods.append("Heuristic")
            detection_method = " + ".join(methods)
        else:
            is_fake = heuristic_score > 35
            confidence = min(99.0, max(1.0, heuristic_score))

        processing_time = round(time.time() - start_time, 2)
        if is_fake:
            top_factors = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:3]
            factors_text = ", ".join([f[0] for f in top_factors if f[1] > 3])
            ml_note = f" [{detection_method}]" if ml_confidence is not None else ""
            explanation = f"{ml_note} Potential manipulation detected. Key indicators: {factors_text}."
        else:
            explanation = f"[{detection_method}] No significant indicators found."

        return {
            "is_fake": bool(is_fake), "confidence": float(round(confidence, 1)),
            "detection_method": detection_method, "frames_analyzed": int(frames_analyzed),
            "faces_detected": int(faces_detected), "ml_frames_scored": int(len(ml_predictions)),
            "explanation": str(explanation), "processing_time": float(processing_time),
            "recommended_action": "Exercise caution" if is_fake else "Authentic",
            "analysis_details": {k: float(v) for k, v in scores.items()}
        }

    def analyze_live_frame(self, frame_bytes: bytes) -> dict:
        import cv2
        import numpy as np
        
        nparr = np.frombuffer(frame_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None: return {"error": "Invalid frame"}
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = []
        if self.face_cascade:
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        if len(faces) == 0: return {"error": "No face detected"}
            
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        (x, y, fw, fh) = faces[0]
        
        pad_w, pad_h = int(fw * 0.1), int(fh * 0.1)
        start_x, start_y = max(0, x - pad_w), max(0, y - pad_h)
        end_x, end_y = min(w, x + fw + pad_w), min(h, y + fh + pad_h)
        
        face_roi = frame[start_y:end_y, start_x:end_x]
        if face_roi.size == 0: return {"error": "Invalid face crop"}
            
        # Mock sequence for live frame (just duplicate)
        sequence = [face_roi] * self.num_segments
        probs = self._predict_sequence(sequence)
        vit_prob, ml_prob = probs.get("vit"), probs.get("efficientnet")
        
        fusion_score = 0.0
        if vit_prob is not None and ml_prob is not None:
            fusion_score = max(ml_prob, vit_prob)
            if abs(ml_prob - vit_prob) > 0.5:
                fusion_score = (ml_prob * 0.4) + (vit_prob * 0.4) + (max(ml_prob, vit_prob) * 0.2)
        elif vit_prob is not None: fusion_score = vit_prob
        elif ml_prob is not None: fusion_score = ml_prob
            
        threshold = 0.45
        is_fake = fusion_score > threshold
        confidence = min(99.0, max(55.0, fusion_score * 100)) if is_fake else min(99.0, max(50.0, (1 - fusion_score) * 100))
            
        return {
            "is_fake": bool(is_fake), "confidence": float(round(confidence, 1)),
            "fusion_score": float(round(fusion_score, 3)), "face_box": {"x": int(x), "y": int(y), "w": int(fw), "h": int(fh)}
        }

deepfake_detector = DeepfakeDetector()
