import os
import logging
logger = logging.getLogger(__name__)
import asyncio
from app.celery_app import celery_app

redis_available = True

from concurrent.futures import ThreadPoolExecutor

def _run_coroutine_in_new_loop(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()

_executor = ThreadPoolExecutor(max_workers=1)

# Wrapper to run async logic inside celery's sync environment
def run_async(coro):
    """Utility to run async functions within celery tasks."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None

    if loop and loop.is_running():
        # Eager execution inside FastAPI/Uvicorn event loop: execute in a separate thread
        future = _executor.submit(_run_coroutine_in_new_loop, coro)
        return future.result()
    else:
        # Normal worker execution: execute in a new loop directly
        return _run_coroutine_in_new_loop(coro)

@celery_app.task(name="detect_deepfake_task")
def detect_deepfake_task(detection_id: str, filepath: str):
    logger.info(f"[CELERY] Starting deepfake inference for {detection_id}")
    from app.core.storage import storage_manager
    local_path = None
    try:
        local_path = run_async(storage_manager.get_file_path(filepath))
        from app.services.detection.deepfake_service import deepfake_detector
        
        # Now update DB
        from app.db.session import async_session
        from sqlalchemy import select
        from app.models.database import Detection, DetectionStatus
        
        # We need to simulate the async task logic.
        result = deepfake_detector.analyze(local_path)
        
        async def update_db():
            async with async_session() as db:
                query = select(Detection).where(Detection.id == detection_id)
                db_res = await db.execute(query)
                detection = db_res.scalar_one_or_none()
                if detection:
                    detection.status = DetectionStatus.COMPLETED
                    detection.is_fake = result["is_fake"]
                    detection.confidence = result["confidence"]
                    detection.explanation = result["explanation"]
                    detection.recommended_action = result.get("recommended_action")
                    detection.processing_time = float(result["processing_time"])
                    
                    # Save metrics into metadata JSON column
                    detection.metadata_ = result.get("analysis_details", {})
                    await db.commit()
        
        run_async(update_db())
        logger.info(f"[CELERY] Completed deepfake inference for {detection_id}")
        return result
    except Exception as e:
        logger.error(f"[CELERY] Deepfake task failed: {e}", exc_info=True)
        # Update DB on failure
        from app.db.session import async_session
        from sqlalchemy import select
        from app.models.database import Detection, DetectionStatus
        async def fail_db():
            async with async_session() as db:
                query = select(Detection).where(Detection.id == detection_id)
                db_res = await db.execute(query)
                detection = db_res.scalar_one_or_none()
                if detection:
                    detection.status = DetectionStatus.FAILED
                    detection.explanation = str(e)[:500]
                    await db.commit()
        run_async(fail_db())
        raise
    finally:
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.info(f"[CELERY] Cleaned up file: {local_path}")
            except Exception as e:
                logger.error(f"[CELERY] Failed to delete file {local_path}: {e}")

@celery_app.task(name="detect_voice_task")
def detect_voice_task(detection_id: str, filepath: str):
    logger.info(f"[CELERY] Starting voice spoofing inference for {detection_id}")
    from app.core.storage import storage_manager
    local_path = None
    try:
        local_path = run_async(storage_manager.get_file_path(filepath))
        from app.services.detection.voice_service import voice_detector
        result = voice_detector.analyze(local_path)
        
        from app.db.session import async_session
        from sqlalchemy import select
        from app.models.database import Detection, DetectionStatus
        
        async def update_db():
            async with async_session() as db:
                query = select(Detection).where(Detection.id == detection_id)
                db_res = await db.execute(query)
                detection = db_res.scalar_one_or_none()
                if detection:
                    detection.status = DetectionStatus.COMPLETED
                    detection.is_fake = result["is_fake"]
                    detection.confidence = result["confidence"]
                    detection.explanation = result["explanation"]
                    detection.recommended_action = result.get("recommended_action")
                    detection.processing_time = float(result["processing_time"])
                    
                    # Save metrics into metadata JSON column
                    detection.metadata_ = result.get("analysis_details", {})
                    await db.commit()
                    
        run_async(update_db())
        return result
    except Exception as e:
        logger.error(f"[CELERY] Voice task failed: {e}", exc_info=True)
        from app.db.session import async_session
        from sqlalchemy import select
        from app.models.database import Detection, DetectionStatus
        async def fail_db():
            async with async_session() as db:
                query = select(Detection).where(Detection.id == detection_id)
                db_res = await db.execute(query)
                detection = db_res.scalar_one_or_none()
                if detection:
                    detection.status = DetectionStatus.FAILED
                    detection.explanation = str(e)[:500]
                    await db.commit()
        run_async(fail_db())
        raise
    finally:
        if local_path and os.path.exists(local_path):
            try:
                os.remove(local_path)
                logger.info(f"[CELERY] Cleaned up file: {local_path}")
            except Exception as e:
                logger.error(f"[CELERY] Failed to delete file {local_path}: {e}")

@celery_app.task(name='safe_test_task')
def safe_test_task(a, b):
    logger.info('Executing safe_test_task')
    return a + b
