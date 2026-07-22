"""
Features:
- SQL database persistence
- NLP-based complaint analysis & auto severity scoring
- Real-time SMTP, SendGrid, and Resend notifications
- Ownership-based access control
- UUID-based identifiers
"""

import html
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime, timezone
import uuid
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import re
from collections import Counter
import httpx
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, desc

from app.db.session import get_db
from app.models.database import User, UserRole, Complaint
from app.core.auth import get_current_user
from app.core.config import settings

logger = logging.getLogger(__name__)
router = APIRouter()

# ─────────────────────────────────────────
# NLP — Keyword Analysis Engine
# ─────────────────────────────────────────

CRITICAL_KEYWORDS = [
    "suicide", "kill", "murder", "rape", "blackmail", "threat", "bomb",
    "attack", "weapon", "nude", "naked", "sextortion", "ransom", "kidnap",
    "assault", "violence", "die", "death"
]

HIGH_KEYWORDS = [
    "fraud", "hack", "stolen", "account", "money", "bank", "otp", "phishing",
    "fake", "scam", "stalk", "harass", "abuse", "intimate", "video", "photo",
    "leaked", "extort", "identity", "impersonate", "deepfake", "morphed"
]

MEDIUM_KEYWORDS = [
    "spam", "suspicious", "unknown", "weird", "uncomfortable", "annoying",
    "following", "watching", "message", "call", "profile", "social media"
]

CYBER_CATEGORIES = {
    "financial_fraud": ["bank", "money", "otp", "upi", "fraud", "transfer", "payment", "account", "stolen", "scam"],
    "deepfake": ["deepfake", "morphed", "fake video", "ai generated", "manipulated", "face swap"],
    "stalking": ["stalk", "follow", "track", "watch", "location", "home", "office"],
    "harassment": ["harass", "bully", "abuse", "threat", "intimidate", "message", "call"],
    "sextortion": ["nude", "naked", "intimate", "photo", "video", "blackmail", "leak", "sextortion"],
    "identity_theft": ["identity", "impersonate", "fake profile", "fake account", "pretend"],
    "phishing": ["phishing", "link", "website", "url", "click", "password", "login"],
}


def analyze_complaint_nlp(description: str, complaint_type: str) -> dict:
    """
    NLP analysis — keyword extraction, severity scoring, category detection
    """
    text = description.lower()
    words = re.findall(r'\b\w+\b', text)

    # Severity scoring
    critical_hits = [w for w in CRITICAL_KEYWORDS if w in text]
    high_hits = [w for w in HIGH_KEYWORDS if w in text]
    medium_hits = [w for w in MEDIUM_KEYWORDS if w in text]

    severity_score = (len(critical_hits) * 10) + (len(high_hits) * 5) + (len(medium_hits) * 2)

    # Auto priority
    if critical_hits or severity_score >= 15:
        auto_priority = "critical"
        suggested_urgency = "high"
    elif high_hits or severity_score >= 8:
        auto_priority = "high"
        suggested_urgency = "high"
    elif medium_hits or severity_score >= 3:
        auto_priority = "medium"
        suggested_urgency = "medium"
    else:
        auto_priority = "low"
        suggested_urgency = "low"

    # Category detection
    detected_categories = []
    for category, keywords in CYBER_CATEGORIES.items():
        if any(kw in text for kw in keywords):
            detected_categories.append(category)

    # Key phrases extraction (simple — top repeated meaningful words)
    stopwords = {"the", "a", "an", "is", "in", "on", "at", "to", "for", "of", "and", "or", "i", "my", "me", "was", "he", "she", "they", "it", "this", "that", "have", "has", "been", "are", "were"}
    meaningful_words = [w for w in words if w not in stopwords and len(w) > 3]
    key_phrases = [w for w, _ in Counter(meaningful_words).most_common(5)]

    # Risk flags
    risk_flags = []
    if critical_hits:
        risk_flags.append(f"⚠️ Critical keywords detected: {', '.join(critical_hits)}")
    if high_hits:
        risk_flags.append(f"🔴 High-risk keywords: {', '.join(high_hits[:3])}")
    if "minor" in text or "child" in text or "underage" in text:
        risk_flags.append("🚨 Possible minor involvement — escalate immediately")
    if any(w in text for w in ["suicide", "kill", "die", "death"]):
        risk_flags.append("🆘 Life-threatening indicators — immediate response required")

    return {
        "severity_score": severity_score,
        "auto_priority": auto_priority,
        "suggested_urgency": suggested_urgency,
        "detected_categories": detected_categories,
        "key_phrases": key_phrases,
        "risk_flags": risk_flags,
        "critical_keywords_found": critical_hits,
        "high_keywords_found": high_hits[:5],
        "word_count": len(words),
        "analysis_confidence": "high" if len(words) > 20 else "medium" if len(words) > 10 else "low"
    }


# ─────────────────────────────────────────
# Email Notification
# ─────────────────────────────────────────

async def send_notification_email(complaint: dict, nlp_result: dict):
    """
    Send email notification to cyber cell officer using SMTP, SendGrid, or Resend
    """
    priority = nlp_result.get("auto_priority", "medium").upper()
    tracking_id = complaint.get("tracking_id", "N/A")
    case_ref = complaint.get("case_ref", "N/A")
    complaint_type = complaint.get("type", "N/A")
    city = complaint.get("city", "N/A")
    risk_flags = nlp_result.get("risk_flags", [])

    notification_content = f"""
╔══════════════════════════════════════════════════╗
║       NovaShield — NEW COMPLAINT ALERT           ║
╚══════════════════════════════════════════════════╝

🚨 PRIORITY: {priority}
📋 Tracking ID: {tracking_id}
📁 Draft Reference ID: {case_ref}
🔍 Type: {complaint_type}
🏙️ City: {city}
⚡ Severity Score: {nlp_result.get('severity_score', 0)}/100
🏷️ Categories: {', '.join(nlp_result.get('detected_categories', ['General']))}
🔑 Key Phrases: {', '.join(nlp_result.get('key_phrases', []))}

Risk Flags:
{chr(10).join(risk_flags) if risk_flags else 'No critical flags'}

Action Required: {"IMMEDIATE RESPONSE" if priority == "CRITICAL" else "Review within 24 hours"}
    """

    # 1. Try SendGrid API
    if settings.SENDGRID_API_KEY:
        try:
            logger.info("Sending email via SendGrid...")
            headers = {
                "Authorization": f"Bearer {settings.SENDGRID_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "personalizations": [{"to": [{"email": settings.SMTP_USER or "officer@cybercell.gov.in"}]}],
                "from": {"email": settings.SENDGRID_FROM_EMAIL or "no-reply@kavach.gov.in"},
                "subject": f"[{priority}] New Cybercrime Complaint Draft — {tracking_id}",
                "content": [{"type": "text/plain", "value": notification_content}]
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post("https://api.sendgrid.com/v3/mail/send", headers=headers, json=payload)
                if resp.status_code == 202:
                    logger.info("Email sent successfully via SendGrid.")
                    return
                else:
                    logger.error(f"SendGrid failed: {resp.text}")
        except Exception as e:
            logger.error(f"SendGrid transmission error: {e}")

    # 2. Try Resend API
    if settings.RESEND_API_KEY:
        try:
            logger.info("Sending email via Resend...")
            headers = {
                "Authorization": f"Bearer {settings.RESEND_API_KEY}",
                "Content-Type": "application/json"
            }
            payload = {
                "from": settings.RESEND_FROM_EMAIL or "no-reply@kavach.gov.in",
                "to": settings.SMTP_USER or "officer@cybercell.gov.in",
                "subject": f"[{priority}] New Cybercrime Complaint Draft — {tracking_id}",
                "text": notification_content
            }
            async with httpx.AsyncClient() as client:
                resp = await client.post("https://api.resend.com/emails", headers=headers, json=payload)
                if resp.status_code == 200:
                    logger.info("Email sent successfully via Resend.")
                    return
                else:
                    logger.error(f"Resend failed: {resp.text}")
        except Exception as e:
            logger.error(f"Resend transmission error: {e}")

    # 3. Try standard SMTP
    if settings.SMTP_HOST and settings.SMTP_USER and settings.SMTP_PASSWORD:
        try:
            logger.info("Sending email via SMTP...")
            msg = MIMEMultipart()
            msg['From'] = settings.SMTP_USER
            msg['To'] = settings.SMTP_USER  # Self-notification
            msg['Subject'] = f"[{priority}] New Cybercrime Complaint Draft — {tracking_id}"
            msg.attach(MIMEText(notification_content, 'plain'))
            
            # Since standard SMTP can be blocking, we run it synchronously in background executor
            import asyncio
            def send_sync():
                with smtplib.SMTP(settings.SMTP_HOST, settings.SMTP_PORT) as server:
                    server.starttls()
                    server.login(settings.SMTP_USER, settings.SMTP_PASSWORD)
                    server.send_message(msg)
            
            await asyncio.to_thread(send_sync)
            logger.info("Email sent successfully via SMTP.")
            return
        except Exception as e:
            logger.error(f"SMTP transmission error: {e}")

    # Fallback log print
    logger.info(f"📧 Notification Logged (No active mail server configured):\n{notification_content}")


# ─────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────

class ComplaintSubmit(BaseModel):
    name: str
    phone: str
    email: Optional[EmailStr] = None
    city: Optional[str] = None
    type: str
    description: str
    evidence: Optional[str] = None
    urgency: str = "medium"
    loss_amount: Optional[str] = None


class ComplaintResponse(BaseModel):
    success: bool
    tracking_id: str
    case_ref: str
    message: str
    submitted_at: str
    auto_priority: str
    severity_score: int
    risk_flags: list
    detected_categories: list


# ─────────────────────────────────────────
# Routes
# ─────────────────────────────────────────

@router.post("/complaints/submit", response_model=ComplaintResponse)
async def submit_complaint(
    complaint: ComplaintSubmit,
    background_tasks: BackgroundTasks,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    try:
        # Generate 100% UUID-based identifiers
        tracking_id = f"CC-DRAFT-{uuid.uuid4()}"
        case_ref = f"CC-REF-{uuid.uuid4()}"
        submitted_at_dt = datetime.now(timezone.utc)
        submitted_at = submitted_at_dt.isoformat()

        # Sanitize and HTML escape inputs to prevent XSS
        safe_description = html.escape(complaint.description)
        safe_name = html.escape(complaint.name)
        safe_phone = html.escape(complaint.phone)
        safe_city = html.escape(complaint.city or "")
        safe_evidence = html.escape(complaint.evidence or "")

        # NLP Analysis
        nlp = analyze_complaint_nlp(safe_description, complaint.type)

        # Override urgency if NLP detects higher severity
        final_urgency = complaint.urgency
        if nlp["auto_priority"] == "critical":
            final_urgency = "high"
        elif nlp["auto_priority"] == "high" and complaint.urgency == "low":
            final_urgency = "medium"

        timeline_data = [
            {"date": submitted_at, "event": "Complaint draft submitted via KAVACH Portal", "icon": "📤", "done": True},
            {"date": submitted_at, "event": f"AI analysis complete — Priority: {nlp['auto_priority'].upper()}", "icon": "🤖", "done": True},
            {"date": "Pending", "event": "Draft validation & official submission", "icon": "📝", "done": False},
        ]

        # Save to DB
        complaint_db = Complaint(
            user_id=current_user.id,
            created_by=current_user.id,
            tracking_id=tracking_id,
            case_ref=case_ref,
            name=safe_name,
            phone=safe_phone,
            email=complaint.email,
            city=safe_city,
            type=complaint.type,
            description=safe_description,
            evidence=safe_evidence,
            urgency=final_urgency,
            status="pending",
            submitted_at=submitted_at_dt,
            last_updated=submitted_at_dt,
            nlp_analysis=nlp,
            timeline=timeline_data
        )

        db.add(complaint_db)
        await db.commit()

        complaint_dict = {
            "tracking_id": tracking_id,
            "case_ref": case_ref,
            "name": safe_name,
            "phone": safe_phone,
            "email": complaint.email,
            "city": safe_city,
            "type": complaint.type,
            "description": safe_description,
            "evidence": safe_evidence,
            "urgency": final_urgency,
            "status": "pending"
        }

        # Send notification in background
        background_tasks.add_task(send_notification_email, complaint_dict, nlp)

        logger.info(f"✅ Complaint {tracking_id} | Priority: {nlp['auto_priority']} | User: {current_user.username}")

        return ComplaintResponse(
            success=True,
            tracking_id=tracking_id,
            case_ref=case_ref,
            message=f"Complaint registered. Tracking ID: {tracking_id}. AI Priority: {nlp['auto_priority'].upper()}.",
            submitted_at=submitted_at,
            auto_priority=nlp["auto_priority"],
            severity_score=nlp["severity_score"],
            risk_flags=nlp["risk_flags"],
            detected_categories=nlp["detected_categories"],
        )

    except Exception as e:
        logger.error(f"Complaint submission failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to submit complaint.")


@router.get("/complaints/track/{tracking_id}")
async def track_complaint(
    tracking_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    result = await db.execute(
        select(Complaint).where(Complaint.tracking_id == tracking_id)
    )
    complaint = result.scalar_one_or_none()
    
    if not complaint:
        raise HTTPException(status_code=404, detail=f"No complaint found: {tracking_id}")

    # Enforce ownership based authorization
    if complaint.user_id != current_user.id and current_user.role != UserRole.ADMIN:
        raise HTTPException(status_code=403, detail="Unauthorized to track this complaint")

    # Serialize complaint
    return {
        "success": True,
        "data": {
            "tracking_id": complaint.tracking_id,
            "case_ref": complaint.case_ref,
            "name": complaint.name,
            "phone": complaint.phone,
            "email": complaint.email,
            "city": complaint.city,
            "type": complaint.type,
            "description": complaint.description,
            "evidence": complaint.evidence,
            "urgency": complaint.urgency,
            "status": complaint.status,
            "submitted_at": complaint.submitted_at.isoformat(),
            "last_updated": complaint.last_updated.isoformat(),
            "nlp_analysis": complaint.nlp_analysis,
            "timeline": complaint.timeline
        }
    }


@router.get("/complaints/all")
async def get_all_complaints(
    db: AsyncSession = Depends(get_db),
    current_user: Optional[User] = Depends(get_current_user) # enforce authentication
):
    """
    Get complaints feed.
    - If Admin: Return full details of all complaints.
    - If Regular User: Return full details of only user's own complaints.
    """
    if current_user.role == UserRole.ADMIN:
        result = await db.execute(select(Complaint).order_by(desc(Complaint.created_at)))
    else:
        result = await db.execute(
            select(Complaint).where(Complaint.user_id == current_user.id).order_by(desc(Complaint.created_at))
        )
    complaints = result.scalars().all()

    items = []
    for c in complaints:
        items.append({
            "tracking_id": c.tracking_id,
            "case_ref": c.case_ref,
            "name": c.name,
            "phone": c.phone,
            "email": c.email,
            "city": c.city,
            "type": c.type,
            "description": c.description,
            "evidence": c.evidence,
            "urgency": c.urgency,
            "status": c.status,
            "submitted_at": c.submitted_at.isoformat(),
            "last_updated": c.last_updated.isoformat(),
            "nlp_analysis": c.nlp_analysis,
            "timeline": c.timeline
        })

    return {
        "success": True,
        "total": len(items),
        "complaints": items
    }


@router.get("/complaints/stats")
async def complaint_stats(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Fetch complaint statistics. Regular users get stats over their own complaints, admins get overall.
    """
    query = select(Complaint)
    if current_user.role != UserRole.ADMIN:
        query = query.where(Complaint.user_id == current_user.id)

    result = await db.execute(query)
    complaints = result.scalars().all()

    total = len(complaints)
    by_type = {}
    by_status = {}
    by_urgency = {}
    by_priority = {}
    by_city = {}
    severity_scores = []

    for c in complaints:
        by_type[c.type] = by_type.get(c.type, 0) + 1
        by_status[c.status] = by_status.get(c.status, 0) + 1
        by_urgency[c.urgency] = by_urgency.get(c.urgency, 0) + 1
        by_city[c.city] = by_city.get(c.city, 0) + 1
        if c.nlp_analysis:
            priority = c.nlp_analysis.get("auto_priority", "medium")
            by_priority[priority] = by_priority.get(priority, 0) + 1
            severity_scores.append(c.nlp_analysis.get("severity_score", 0))

    avg_severity = sum(severity_scores) / len(severity_scores) if severity_scores else 0

    return {
        "success": True,
        "total": total,
        "by_type": by_type,
        "by_status": by_status,
        "by_urgency": by_urgency,
        "by_priority": by_priority,
        "by_city": by_city,
        "avg_severity_score": round(avg_severity, 1),
        "high_priority_count": by_priority.get("critical", 0) + by_priority.get("high", 0),
    }


@router.get("/complaints/analytics/trends")
async def complaint_trends(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """
    Trend detection - patterns over last 7 days.
    """
    now = datetime.now(timezone.utc)
    
    # Query complaints. Non-admins only query their own trends.
    query = select(Complaint)
    if current_user.role != UserRole.ADMIN:
        query = query.where(Complaint.user_id == current_user.id)
        
    result = await db.execute(query)
    complaints = result.scalars().all()

    daily_counts = {}
    category_trends = {}
    city_hotspots = {}

    for c in complaints:
        try:
            submitted = c.submitted_at.replace(tzinfo=timezone.utc)
            days_ago = (now - submitted).days
            if days_ago <= 7:
                day_key = submitted.strftime("%a %d %b")
                daily_counts[day_key] = daily_counts.get(day_key, 0) + 1

                # Category trends
                if c.nlp_analysis and c.nlp_analysis.get("detected_categories"):
                    for cat in c.nlp_analysis["detected_categories"]:
                        category_trends[cat] = category_trends.get(cat, 0) + 1

                # City hotspots
                city = c.city or "Unknown"
                city_hotspots[city] = city_hotspots.get(city, 0) + 1
        except Exception:
            pass

    # Prediction — simple trend
    counts = list(daily_counts.values())
    if len(counts) >= 3:
        recent_avg = sum(counts[-3:]) / 3
        older_avg = sum(counts[:-3]) / max(len(counts[:-3]), 1)
        trend = "increasing" if recent_avg > older_avg else "decreasing" if recent_avg < older_avg else "stable"
        predicted_tomorrow = round(recent_avg * (1.1 if trend == "increasing" else 0.9))
    else:
        trend = "insufficient_data"
        predicted_tomorrow = 0

    top_category = max(category_trends, key=category_trends.get) if category_trends else "N/A"
    top_city = max(city_hotspots, key=city_hotspots.get) if city_hotspots else "N/A"

    return {
        "success": True,
        "period": "Last 7 days",
        "daily_counts": daily_counts,
        "total_this_week": sum(daily_counts.values()),
        "trend": trend,
        "predicted_tomorrow": predicted_tomorrow,
        "top_category": top_category,
        "top_city_hotspot": top_city,
        "category_breakdown": category_trends,
        "city_hotspots": city_hotspots,
        "alert": f"⚠️ {top_category.replace('_', ' ').title()} complaints are {trend} — monitor closely" if trend == "increasing" else None
    }
