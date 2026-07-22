"""
KAVACH AI Pro - Admin Endpoints
System statistics, user management, and audit logs
"""

from typing import Optional
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.db.session import get_db
from app.models.database import User, Detection, AuditLog, UserRole, DetectionType, DetectionStatus
from app.core.auth import get_current_admin

# Guard the entire router with Admin role check
router = APIRouter(dependencies=[Depends(get_current_admin)])


@router.get("/stats")
async def get_system_stats(db: AsyncSession = Depends(get_db)):
    """Get system-wide statistics"""
    total_users = (await db.execute(
        select(func.count()).select_from(User)
    )).scalar() or 0

    total_detections = (await db.execute(
        select(func.count()).select_from(Detection)
    )).scalar() or 0

    total_threats = (await db.execute(
        select(func.count()).select_from(Detection).where(Detection.is_fake == True)
    )).scalar() or 0

    safe_content = total_detections - total_threats

    completed = (await db.execute(
        select(func.count()).select_from(Detection).where(
            Detection.status == DetectionStatus.COMPLETED
        )
    )).scalar() or 0
    detection_rate = (completed / total_detections * 100) if total_detections > 0 else 0

    # By type
    type_counts = {}
    for det_type in DetectionType:
        count = (await db.execute(
            select(func.count()).select_from(Detection).where(
                Detection.detection_type == det_type
            )
        )).scalar() or 0
        type_counts[det_type.value] = count

    # Recent activity
    recent_result = await db.execute(
        select(Detection).order_by(desc(Detection.created_at)).limit(10)
    )
    recent = recent_result.scalars().all()
    recent_activity = [
        {
            "id": d.id,
            "type": d.detection_type.value if d.detection_type else "unknown",
            "status": d.status.value if d.status else "unknown",
            "is_fake": d.is_fake,
            "confidence": d.confidence,
            "created_at": d.created_at.isoformat() if d.created_at else None
        }
        for d in recent
    ]

    return {
        "total_users": total_users,
        "total_detections": total_detections,
        "total_threats": total_threats,
        "safe_content": safe_content,
        "detection_rate": round(detection_rate, 1),
        "detections_by_type": type_counts,
        "recent_activity": recent_activity
    }


@router.get("/users")
async def list_users(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
):
    """List all users"""
    total = (await db.execute(select(func.count()).select_from(User))).scalar() or 0

    result = await db.execute(
        select(User).order_by(desc(User.created_at))
        .offset((page - 1) * per_page).limit(per_page)
    )
    users = result.scalars().all()

    return {
        "items": [
            {
                "id": u.id,
                "email": u.email,
                "username": u.username,
                "full_name": u.full_name,
                "role": u.role.value if u.role else "user",
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat() if u.created_at else None,
                "last_login": u.last_login.isoformat() if u.last_login else None,
            }
            for u in users
        ],
        "total": total,
        "page": page,
        "per_page": per_page
    }


@router.get("/audit-logs")
async def get_audit_logs(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    action: Optional[str] = Query(None),
    db: AsyncSession = Depends(get_db),
):
    """View audit logs"""
    query = select(AuditLog)
    if action:
        query = query.where(AuditLog.action == action)

    total = (await db.execute(
        select(func.count()).select_from(query.subquery())
    )).scalar() or 0

    result = await db.execute(
        query.order_by(desc(AuditLog.created_at))
        .offset((page - 1) * per_page).limit(per_page)
    )
    logs = result.scalars().all()

    items = [
        {
            "id": log.id,
            "user_id": log.user_id,
            "action": log.action,
            "resource_type": log.resource_type,
            "resource_id": log.resource_id,
            "ip_address": log.ip_address,
            "details": log.details,
            "created_at": log.created_at.isoformat() if log.created_at else None,
        }
        for log in logs
    ]

    return {
        "items": items,
        "logs": items,          # alias for frontend compatibility
        "total": total,
        "page": page,
        "per_page": per_page
    }


class RoleUpdateRequest(BaseModel):
    role: str


@router.put("/users/{user_id}/role")
async def update_user_role(
    user_id: str,
    body: RoleUpdateRequest,
    db: AsyncSession = Depends(get_db),
):
    """Change a user's role (admin only)"""
    try:
        role_enum = UserRole(body.role)
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid role '{body.role}'. Must be one of: user, premium, admin")

    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    user.role = role_enum
    await db.commit()

    return {
        "success": True,
        "message": f"Role updated to '{body.role}' for user {user.username or user.email}",
        "user_id": user_id,
        "new_role": body.role,
    }
