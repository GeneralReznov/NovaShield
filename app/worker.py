import os
import logging
logger = logging.getLogger(__name__)
import asyncio
from concurrent.futures import ThreadPoolExecutor

# ---------------------------------------------------------------------------
# Celery is optional — if Redis is unavailable the app still starts normally.
# Inference is handled by FastAPI BackgroundTasks; Celery tasks are kept here
# only for deployments that explicitly configure a broker.
# ---------------------------------------------------------------------------
celery_app = None
_celery_available = False

try:
    from app.celery_app import celery_app as _celery_app
    celery_app = _celery_app
    _celery_available = True
    logger.info("Celery loaded successfully.")
except Exception as e:
    logger.warning(f"Celery not available (Redis may not be configured): {e}. "
                   "Running in BackgroundTasks-only mode.")


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
        future = _executor.submit(_run_coroutine_in_new_loop, coro)
        return future.result()
    else:
        return _run_coroutine_in_new_loop(coro)


def _make_celery_task(name):
    """Return a no-op stub when Celery is not available."""
    def stub(*args, **kwargs):
        raise RuntimeError(
            f"Celery task '{name}' called but Celery/Redis is not configured. "
            "Use FastAPI BackgroundTasks instead."
        )
    stub.__name__ = name
    return stub


if _celery_available and celery_app is not None:
    @celery_app.task(name="detect_deepfake_task")
    def detect_deepfake_task(detection_id: str, filepath: str):
        logger.info(f"[CELERY] Starting deepfake inference for {detection_id}")
        from app.core.storage import storage_manager
        local_path = None
        try:
            local_path = run_async(storage_manager.get_file_path(filepath))
            from app.services.detection.deepfake_service import deepfake_detector
            from app.db.session import async_session
            from sqlalchemy import select
            from app.models.database import Detection, DetectionStatus

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
                        detection.metadata_ = result.get("analysis_details", {})
                        await db.commit()

            run_async(update_db())
            logger.info(f"[CELERY] Completed deepfake inference for {detection_id}")
            return result
        except Exception as e:
            logger.error(f"[CELERY] Deepfake task failed: {e}", exc_info=True)
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
                except Exception as exc:
                    logger.error(f"[CELERY] Failed to delete file {local_path}: {exc}")

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
                except Exception as exc:
                    logger.error(f"[CELERY] Failed to delete file {local_path}: {exc}")

    @celery_app.task(name='safe_test_task')
    def safe_test_task(a, b):
        logger.info('Executing safe_test_task')
        return a + b

else:
    # Stub functions so any code that references these names at import time
    # won't crash — they will raise clearly at call time.
    detect_deepfake_task = _make_celery_task("detect_deepfake_task")
    detect_voice_task = _make_celery_task("detect_voice_task")
    safe_test_task = _make_celery_task("safe_test_task")
