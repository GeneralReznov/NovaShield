"""
Asynchronous background AI processing with timeout protection and cleanup
"""

import os
import aiofiles
import uuid
import logging
import html
import time
from datetime import datetime, timezone
from typing import Optional
import hashlib

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Query, Request, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse
from fastapi.concurrency import run_in_threadpool
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.db.session import get_db, async_session
from app.models.database import User, UserRole, Detection, DetectionType, DetectionStatus
from app.core.config import settings
from app.core.storage import storage_manager
from app.core.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter()

# Size limits
MAX_VIDEO_SIZE = 50 * 1024 * 1024  # 50MB
MAX_AUDIO_SIZE = 20 * 1024 * 1024  # 20MB
MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB


def _verify_file_signature(chunk: bytes, ext: str) -> bool:
    """Verify that the first chunk's magic bytes match the extension.
    Returns True if valid, False otherwise.
    """
    if len(chunk) < 4:
        return False

    ext = ext.lower()

    # 1. Video signatures
    if ext == '.mp4':
        return len(chunk) >= 8 and chunk[4:8] == b'ftyp'
    elif ext == '.avi':
        return chunk.startswith(b'RIFF') and len(chunk) >= 12 and chunk[8:12] == b'AVI '
    elif ext in ('.mkv', '.webm'):
        return chunk.startswith(b'\x1A\x45\xDF\xA3')
    elif ext == '.mov':
        return (len(chunk) >= 8 and chunk[4:8] in (b'ftyp', b'moov', b'free', b'mdat', b'wide')) or chunk.startswith(b'\x00\x00\x00')

    # 2. Audio signatures
    elif ext == '.wav':
        return chunk.startswith(b'RIFF') and len(chunk) >= 12 and chunk[8:12] == b'WAVE'
    elif ext == '.mp3':
        # ID3 tag or MP3 frame sync (first 11 bits are set to 1)
        return chunk.startswith(b'ID3') or (len(chunk) >= 2 and chunk[0] == 0xFF and (chunk[1] & 0xE0) == 0xE0)
    elif ext == '.m4a':
        return len(chunk) >= 8 and chunk[4:8] == b'ftyp'
    elif ext == '.flac':
        return chunk.startswith(b'fLaC')
    elif ext == '.ogg':
        return chunk.startswith(b'OggS')

    # 3. Image signatures
    elif ext == '.png':
        return chunk.startswith(b'\x89PNG\r\n\x1a\n')
    elif ext in ('.jpg', '.jpeg'):
        return chunk.startswith(b'\xFF\xD8\xFF')

    return False  # Default to False for unhandled extensions for security


async def _save_upload(file: UploadFile, detection_type: str) -> tuple[str, int, str]:
    """Save uploaded file to disk and return (path, size, sha256_hash).
    Handles large files by reading in chunks and writing asynchronously.
    Enforces maximum size limits on-the-fly to prevent memory and disk exhaustion.
    Calculates SHA-256 hash on-the-fly for evidence integrity.
    """
    upload_dir = os.path.join(settings.UPLOAD_DIR, detection_type)
    os.makedirs(upload_dir, exist_ok=True)

    # Sanitize file name to prevent directory traversal
    original_name = os.path.basename(file.filename or 'upload.bin')
    ext = os.path.splitext(original_name)[1].lower()
    if ext not in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.ogg', '.wav', '.mp3', '.m4a', '.flac', '.png', '.jpg', '.jpeg']:
        raise ValueError("Unsupported file extension")

    filename = f"{uuid.uuid4().hex}{ext}"
    filepath = os.path.join(upload_dir, filename)

    # Enforce size limits dynamically based on detection type
    if detection_type == "deepfake":
        max_size = MAX_VIDEO_SIZE
    elif detection_type == "voice":
        max_size = MAX_AUDIO_SIZE
    else:
        max_size = MAX_IMAGE_SIZE

    # Reset stream
    await file.seek(0)

    total_written = 0
    CHUNK_SIZE = 2 * 1024 * 1024  # 2MB
    first_chunk = True
    hasher = hashlib.sha256()
    
    try:
        async with aiofiles.open(filepath, 'wb') as f:
            while True:
                chunk = await file.read(CHUNK_SIZE)
                if not chunk:
                    break
                if first_chunk:
                    if not _verify_file_signature(chunk, ext):
                        raise ValueError(f"File contents do not match extension '{ext}'")
                    first_chunk = False
                total_written += len(chunk)
                if total_written > max_size:
                    raise ValueError(
                        f"File too large. Max limit is {max_size // (1024 * 1024)}MB."
                    )
                hasher.update(chunk)
                await f.write(chunk)
    except Exception as e:
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
            except Exception:
                pass
        raise e

    # Upload to scalable storage (e.g. S3) if configured
    uri = await storage_manager.save_from_path(filepath, filepath)
    
    # Clean up local file if it was successfully moved to S3
    if uri.startswith("s3://") and os.path.exists(filepath):
        try:
            os.remove(filepath)
        except Exception:
            pass

    if total_written == 0:
        if os.path.exists(filepath):
            os.remove(filepath)
        raise ValueError("Empty file uploaded")

    file_hash = hasher.hexdigest()
    return uri, total_written, file_hash


# Background inference tasks
async def run_deepfake_inference(detection_id: str, filepath: str):
    start_time = time.time()
    logger.info(f"Starting background deepfake inference for detection_id: {detection_id}")
    try:
        from app.services.detection.deepfake_service import deepfake_detector
        
        # Run with timeout protection
        result = await asyncio_wait_with_timeout(
            run_in_threadpool(deepfake_detector.analyze, filepath),
            timeout=settings.INFERENCE_TIMEOUT
        )
        logger.info(f"Deepfake inference completed for {detection_id}. Result keys: {list(result.keys()) if result else None}")

        async with async_session() as db:
            logger.info(f"Querying database for detection_id: {detection_id}")
            result_db = await db.execute(select(Detection).where(Detection.id == detection_id))
            detection = result_db.scalar_one_or_none()
            if detection:
                logger.info(f"Found detection record for {detection_id}. Updating status to COMPLETED.")
                detection.status = DetectionStatus.COMPLETED
                detection.is_fake = result.get('is_fake', False)
                detection.confidence = result.get('confidence', 0.0)
                detection.explanation = result.get('explanation', 'AI analysis complete')
                detection.recommended_action = result.get('recommended_action', 'N/A')
                detection.processing_time = float(time.time() - start_time)
                detection.completed_at = datetime.now(timezone.utc)
                
                # Merge existing metadata with new analysis details to keep sha256_hash
                existing_meta = detection.metadata_ or {}
                new_meta = result.get('analysis_details', {})
                existing_meta.update(new_meta)
                detection.metadata_ = existing_meta
                
                await db.commit()
                logger.info(f"Successfully committed COMPLETED status for {detection_id}")
            else:
                logger.error(f"Detection record NOT found in database for detection_id: {detection_id}")
    except Exception as e:
        logger.error(f"Deepfake background task failed for {detection_id}: {e}", exc_info=True)
        try:
            async with async_session() as db:
                result_db = await db.execute(select(Detection).where(Detection.id == detection_id))
                detection = result_db.scalar_one_or_none()
                if detection:
                    detection.status = DetectionStatus.FAILED
                    detection.explanation = f"Inference failed or timed out: {str(e)[:200]}"
                    detection.completed_at = datetime.now(timezone.utc)
                    await db.commit()
                    logger.info(f"Successfully committed FAILED status for {detection_id}")
                else:
                    logger.error(f"Detection record not found to write FAILED status for {detection_id}")
        except Exception as db_err:
            logger.error(f"Failed to write FAILED status for {detection_id}: {db_err}", exc_info=True)
    finally:
        # DO NOT DELETE file locally if it's the original evidence and we don't have S3 set up
        # If the file path starts with s3://, it was uploaded successfully, local was already cleaned.
        # If it's a local file, we keep it in the uploads folder as evidence (vault feature).
        pass


async def run_voice_inference(detection_id: str, filepath: str):
    start_time = time.time()
    logger.info(f"Starting background voice inference for detection_id: {detection_id}")
    try:
        from app.services.detection.voice_service import voice_detector
        
        result = await asyncio_wait_with_timeout(
            run_in_threadpool(voice_detector.analyze, filepath),
            timeout=settings.INFERENCE_TIMEOUT
        )
        logger.info(f"Voice inference completed for {detection_id}. Result keys: {list(result.keys()) if result else None}")

        async with async_session() as db:
            logger.info(f"Querying database for detection_id: {detection_id}")
            result_db = await db.execute(select(Detection).where(Detection.id == detection_id))
            detection = result_db.scalar_one_or_none()
            if detection:
                logger.info(f"Found detection record for {detection_id}. Updating status to COMPLETED.")
                detection.status = DetectionStatus.COMPLETED
                detection.is_fake = result.get('is_fake', False)
                detection.confidence = result.get('confidence', 0.0)
                detection.spoof_type = result.get('spoof_type', 'Synthetic')
                detection.explanation = result.get('explanation', 'Voice spoofing analysis complete')
                detection.recommended_action = result.get('recommended_action', 'N/A')
                detection.processing_time = float(time.time() - start_time)
                detection.completed_at = datetime.now(timezone.utc)
                
                # Merge existing metadata with new analysis details to keep sha256_hash
                existing_meta = detection.metadata_ or {}
                new_meta = result.get('analysis_details', {})
                existing_meta.update(new_meta)
                detection.metadata_ = existing_meta
                
                await db.commit()
                logger.info(f"Successfully committed COMPLETED status for {detection_id}")
            else:
                logger.error(f"Detection record NOT found in database for detection_id: {detection_id}")
    except Exception as e:
        logger.error(f"Voice background task failed for {detection_id}: {e}", exc_info=True)
        try:
            async with async_session() as db:
                result_db = await db.execute(select(Detection).where(Detection.id == detection_id))
                detection = result_db.scalar_one_or_none()
                if detection:
                    detection.status = DetectionStatus.FAILED
                    detection.explanation = f"Inference failed or timed out: {str(e)[:200]}"
                    detection.completed_at = datetime.now(timezone.utc)
                    await db.commit()
                    logger.info(f"Successfully committed FAILED status for {detection_id}")
                else:
                    logger.error(f"Detection record not found to write FAILED status for {detection_id}")
        except Exception as db_err:
            logger.error(f"Failed to write FAILED status for {detection_id}: {db_err}", exc_info=True)
    finally:
        # DO NOT DELETE file locally to preserve Evidence Chain of Custody
        pass


async def asyncio_wait_with_timeout(coro, timeout):
    import asyncio
    return await asyncio.wait_for(coro, timeout=timeout)


# ─────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────

@router.post("/detect/deepfake")
async def detect_deepfake(
    background_tasks: BackgroundTasks,
    video: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload a video for background deepfake detection"""
    # Enforce maximum size early
    if video.size is not None and video.size > MAX_VIDEO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {MAX_VIDEO_SIZE // (1024*1024)}MB."
        )

    try:
        filepath, saved_size, file_hash = await _save_upload(video, "deepfake")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create detection record
    detection = Detection(
        user_id=current_user.id,
        detection_owner=current_user.email,
        detection_type=DetectionType.DEEPFAKE,
        status=DetectionStatus.PROCESSING,
        original_filename=html.escape(video.filename or "video.mp4"),
        file_path=filepath,
        file_size=saved_size,
        mime_type=video.content_type,
        metadata_={"sha256_hash": file_hash, "original_filename": video.filename},
    )
    db.add(detection)
    await db.commit()
    detection_id = detection.id

    # Queue background task using FastAPI BackgroundTasks (bypassing Celery to avoid timezone drift on Windows)
    background_tasks.add_task(run_deepfake_inference, detection_id, filepath)

    return {
        "success": True,
        "detection_id": detection_id,
        "status": "processing",
        "message": "Deepfake analysis queued in background"
    }


@router.post("/detect/voice")
async def detect_voice(
    background_tasks: BackgroundTasks,
    audio: UploadFile = File(None),
    voice: UploadFile = File(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Upload audio for background voice anti-spoofing analysis"""
    file = audio or voice
    if not file:
        raise HTTPException(status_code=422, detail="Please upload an audio file")

    if file.size is not None and file.size > MAX_AUDIO_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum is {MAX_AUDIO_SIZE // (1024*1024)}MB."
        )

    try:
        filepath, saved_size, file_hash = await _save_upload(file, "voice")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Create detection record
    detection = Detection(
        user_id=current_user.id,
        detection_owner=current_user.email,
        detection_type=DetectionType.VOICE,
        status=DetectionStatus.PROCESSING,
        original_filename=html.escape(file.filename or "audio.wav"),
        file_path=filepath,
        file_size=saved_size,
        mime_type=file.content_type,
        metadata_={"sha256_hash": file_hash, "original_filename": file.filename},
    )
    db.add(detection)
    await db.commit()
    detection_id = detection.id

    # Queue background task using FastAPI BackgroundTasks
    background_tasks.add_task(run_voice_inference, detection_id, filepath)

    return {
        "success": True,
        "detection_id": detection_id,
        "status": "processing",
        "message": "Voice analysis queued in background"
    }


@router.post("/detect/phishing")
async def detect_phishing(
    request: Request,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Analyze a URL for phishing using XGBoost ML model (synchronous since it takes <1s)"""
    content_type = request.headers.get("content-type", "")
    if "application/json" in content_type:
        body = await request.json()
        url = body.get("url", "")
        content = body.get("content", "")
    else:
        form = await request.form()
        url = form.get("url", "")
        content = form.get("content", "")

    if not url:
        raise HTTPException(status_code=400, detail="Please provide a URL")

    # Clean input
    safe_url = html.escape(url)
    safe_content = html.escape(content or "")

    # Create detection record
    detection = Detection(
        user_id=current_user.id,
        detection_owner=current_user.email,
        detection_type=DetectionType.PHISHING,
        status=DetectionStatus.PROCESSING,
        original_filename=safe_url[:500],
    )
    db.add(detection)
    await db.commit()
    detection_id = detection.id

    try:
        from app.services.detection.phishing_service import phishing_detector
        result = await run_in_threadpool(phishing_detector.analyze, safe_url, safe_content or None)

        detection.status = DetectionStatus.COMPLETED
        detection.is_fake = result.get('is_phishing', False)
        detection.confidence = result.get('confidence', 0.0)
        detection.explanation = result.get('explanation', '')
        detection.recommended_action = result.get('recommended_action', '')
        detection.processing_time = result.get('processing_time', 0.0)
        detection.completed_at = datetime.now(timezone.utc)
        detection.metadata_ = result.get('analysis_details', {})
        await db.commit()

        return {
            "id": detection_id,
            "detection_id": detection_id,
            "status": "completed",
            "is_fake": result.get('is_phishing', False),
            "is_phishing": result.get('is_phishing', False),
            "confidence": result.get('confidence', 0.0),
            "detection_type": "phishing",
            "url": safe_url,
            "threats": result.get('threats', []),
            "risk_level": result.get('risk_level', 'Unknown'),
            "detection_method": result.get('detection_method', 'Heuristic'),
            "explanation": result.get('explanation', ''),
            "recommended_action": result.get('recommended_action', ''),
            "processing_time": result.get('processing_time', 0.0),
            "analysis_details": result.get('analysis_details', {}),
        }
    except Exception as e:
        logger.error(f"Phishing detection failed: {e}", exc_info=True)
        detection.status = DetectionStatus.FAILED
        detection.explanation = str(e)[:500]
        await db.commit()
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)[:200]}")


@router.get("/detections/{detection_id}")
async def get_detection(
    detection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get detection result by ID (restricted to owner or admin)"""
    result = await db.execute(
        select(Detection).where(Detection.id == detection_id)
    )
    detection = result.scalar_one_or_none()

    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    # Enforce ownership
    if detection.user_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Unauthorized to view this record")

    return {
        "id": detection.id,
        "detection_type": detection.detection_type.value if detection.detection_type else None,
        "status": detection.status.value if detection.status else None,
        "is_fake": detection.is_fake,
        "confidence": detection.confidence,
        "explanation": detection.explanation,
        "recommended_action": detection.recommended_action,
        "processing_time": detection.processing_time,
        "original_filename": detection.original_filename,
        "created_at": detection.created_at.isoformat() if detection.created_at else None,
        "completed_at": detection.completed_at.isoformat() if detection.completed_at else None,
        "metadata": detection.metadata_,
        "analysis_details": detection.metadata_,
        "spoof_type": detection.spoof_type,
    }


@router.get("/history")
async def get_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    detection_type: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get paginated detection history (restricted to owner or admin)"""
    query = select(Detection)

    # Restrict to current user unless admin
    if current_user.role != UserRole.ADMIN:
        query = query.where(Detection.user_id == current_user.id)

    if detection_type:
        query = query.where(Detection.detection_type == detection_type)

    # Count total
    count_query = select(func.count()).select_from(query.subquery())
    total = (await db.execute(count_query)).scalar() or 0

    # Fetch paginated results
    query = query.order_by(desc(Detection.created_at))
    query = query.offset((page - 1) * per_page).limit(per_page)
    result = await db.execute(query)
    detections = result.scalars().all()

    items = []
    for d in detections:
        items.append({
            "id": d.id,
            "detection_type": d.detection_type.value if d.detection_type else None,
            "type": d.detection_type.value if d.detection_type else None,
            "status": d.status.value if d.status else None,
            "is_fake": d.is_fake,
            "confidence": d.confidence,
            "explanation": d.explanation,
            "recommended_action": d.recommended_action,
            "processing_time": d.processing_time,
            "original_filename": d.original_filename,
            "created_at": d.created_at.isoformat() if d.created_at else None,
            "completed_at": d.completed_at.isoformat() if d.completed_at else None,
            "metadata": d.metadata_,
            "analysis_details": d.metadata_,
            "spoof_type": d.spoof_type,
        })

    return {
        "items": items,
        "total": total,
        "page": page,
        "per_page": per_page,
        "has_next": (page * per_page) < total
    }


@router.get("/stats")
async def get_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """Get detection statistics (restricted to owner or admin)"""
    query_total = select(func.count(Detection.id))
    query_threats = select(func.count(Detection.id)).where(Detection.is_fake == True)
    query_completed = select(func.count(Detection.id)).where(Detection.status == DetectionStatus.COMPLETED)

    # Restrict to user unless admin
    if current_user.role != UserRole.ADMIN:
        query_total = query_total.where(Detection.user_id == current_user.id)
        query_threats = query_threats.where(Detection.user_id == current_user.id)
        query_completed = query_completed.where(Detection.user_id == current_user.id)

    total = (await db.execute(query_total)).scalar() or 0
    threats = (await db.execute(query_threats)).scalar() or 0
    safe = total - threats
    completed = (await db.execute(query_completed)).scalar() or 0

    return {
        "total_scans": total,
        "threats": threats,
        "safe": safe,
        "success_rate": round((completed / total * 100), 1) if total > 0 else 0.0,
    }


@router.websocket("/detect/live")
async def websocket_live_detection(websocket: WebSocket):
    """
    Real-time Deepfake Detection over WebSocket.
    Expects binary frames or base64 strings and returns instant JSON predictions.
    """
    await websocket.accept()
    logger.info("WebSocket connection established for Live Detection")
    
    from app.services.detection.deepfake_service import deepfake_detector
    import base64
    
    try:
        while True:
            data = await websocket.receive_text()
            
            # The frontend sends: "data:image/jpeg;base64,..."
            if "," in data:
                b64_str = data.split(",")[1]
            else:
                b64_str = data
                
            # Add padding if necessary
            b64_str += "=" * ((4 - len(b64_str) % 4) % 4)
                
            try:
                frame_bytes = base64.b64decode(b64_str)
            except Exception as e:
                await websocket.send_json({"error": "Invalid base64 encoding"})
                continue
                
            # Run lightning-fast prediction
            result = await run_in_threadpool(deepfake_detector.analyze_live_frame, frame_bytes)
            
            # Send result back instantly
            await websocket.send_json(result)
            
    except WebSocketDisconnect:
        logger.info("Live Detection WebSocket disconnected by client")
    except Exception as e:
        logger.error(f"WebSocket error in live detection: {e}", exc_info=True)
        try:
            await websocket.close()
        except:
            pass

@router.get("/detections/{detection_id}/report")
async def generate_secure_report(
    detection_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    from fastapi.responses import HTMLResponse
    import hmac
    
    result = await db.execute(select(Detection).where(Detection.id == detection_id))
    detection = result.scalar_one_or_none()
    
    if not detection:
        raise HTTPException(status_code=404, detail="Detection not found")

    if detection.user_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Unauthorized")
        
    meta = detection.metadata_ or {}
    sha256 = meta.get("sha256_hash", "NOT_GENERATED")
    
    # Generate a digital signature to prove this report was generated by KAVACH AI backend
    signature = hmac.new(
        settings.SECRET_KEY.encode(), 
        f"{detection.id}{sha256}{detection.is_fake}".encode(), 
        hashlib.sha256
    ).hexdigest()
    
    status_text = "THREAT DETECTED (DEEPFAKE)" if detection.is_fake else "SAFE (AUTHENTIC)"
    status_color = "#e53e3e" if detection.is_fake else "#38a169"
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>NovaShield - Secure Digital Evidence Report</title>
        <style>
            body {{ font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; margin: 40px; color: #333; }}
            .header {{ border-bottom: 2px solid #2b6cb0; padding-bottom: 10px; margin-bottom: 20px; }}
            .logo {{ font-size: 24px; font-weight: bold; color: #2b6cb0; }}
            .tagline {{ font-size: 14px; color: #718096; }}
            .report-title {{ text-align: center; font-size: 20px; margin: 30px 0; text-transform: uppercase; letter-spacing: 2px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 30px; }}
            th, td {{ padding: 12px; border: 1px solid #e2e8f0; text-align: left; }}
            th {{ background-color: #f7fafc; width: 30%; font-weight: 600; }}
            .status-box {{ background-color: {status_color}; color: white; padding: 15px; text-align: center; font-size: 18px; font-weight: bold; border-radius: 4px; margin-bottom: 30px; }}
            .footer {{ margin-top: 50px; font-size: 12px; color: #718096; border-top: 1px solid #e2e8f0; padding-top: 20px; }}
            .signature-box {{ background-color: #f7fafc; padding: 10px; font-family: monospace; word-break: break-all; border: 1px dashed #cbd5e0; margin-top: 10px; }}
            @media print {{
                body {{ margin: 0; }}
                button {{ display: none; }}
            }}
        </style>
    </head>
    <body>
        <div class="header">
            <div class="logo">🛡️ NOVASHIELD</div>
            <div class="tagline">Official Digital Evidence Report - Section 65B Compliant</div>
        </div>
        
        <div class="report-title">Forensic AI Analysis Report</div>
        
        <div class="status-box">
            CONCLUSION: {status_text}
        </div>
        
        <table>
            <tr>
                <th>Report ID</th>
                <td>{detection.id}</td>
            </tr>
            <tr>
                <th>Analysis Date</th>
                <td>{detection.completed_at.strftime('%Y-%m-%d %H:%M:%S UTC') if detection.completed_at else 'N/A'}</td>
            </tr>
            <tr>
                <th>Scan Type</th>
                <td>{detection.detection_type.value.upper() if detection.detection_type else 'N/A'}</td>
            </tr>
            <tr>
                <th>Original Filename</th>
                <td>{meta.get('original_filename', detection.original_filename)}</td>
            </tr>
            <tr>
                <th>SHA-256 Digital Fingerprint</th>
                <td style="font-family: monospace; font-size: 13px;">{sha256}</td>
            </tr>
            <tr>
                <th>AI Confidence Score</th>
                <td>{detection.confidence * 100:.2f}%</td>
            </tr>
            <tr>
                <th>Explanation</th>
                <td>{detection.explanation}</td>
            </tr>
        </table>
        
        <div class="footer">
            <p><strong>Digital Signature Validation (HMAC-SHA256):</strong></p>
            <div class="signature-box">{signature}</div>
            <p><em>This report is generated securely by the NovaShield backend server. The SHA-256 fingerprint uniquely identifies the uploaded evidence file. Any modification to the original file will result in a different hash.</em></p>
            <p style="margin-top: 30px; text-align: center;">
                <button onclick="window.print()" style="padding: 10px 20px; background-color: #2b6cb0; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 16px;">Print / Save as PDF</button>
            </p>
        </div>
    </body>
    </html>
    """
    
    return HTMLResponse(content=html_content, status_code=200)
