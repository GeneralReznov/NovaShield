"""
KAVACH AI - Groq AI Integration
Handles FIR drafting, complaint analysis, STT (Whisper), and TTS
"""

import os
import tempfile
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional
from groq import Groq

router = APIRouter()

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY not configured")
    return Groq(api_key=api_key)


# ── Schemas ────────────────────────────────────────────────────────────────────

class FIRDraftRequest(BaseModel):
    complainant_name: str
    complaint_type: str
    incident_date: str
    incident_location: str
    description: str
    accused_details: Optional[str] = ""
    loss_amount: Optional[str] = ""

class ComplaintAnalyzeRequest(BaseModel):
    complaint_type: str
    description: str
    evidence: Optional[str] = ""

class ChatRequest(BaseModel):
    message: str
    context: Optional[str] = ""


# ── Endpoints ──────────────────────────────────────────────────────────────────

@router.post("/groq/fir-draft")
async def generate_fir_draft(req: FIRDraftRequest):
    """Generate a legally-formatted FIR draft using Groq LLM"""
    client = get_groq_client()

    ipc_map = {
        "deepfake": "Sections 66C, 66D IT Act 2000; Section 500 IPC (Defamation); Section 354C IPC",
        "phishing": "Sections 66, 66C, 66D IT Act 2000; Section 420 IPC (Cheating)",
        "voice_spoofing": "Sections 66D IT Act 2000; Section 419 IPC (Cheating by personation)",
        "financial_fraud": "Sections 420, 406 IPC; Sections 66C, 66D IT Act 2000",
        "cyberbullying": "Sections 66A, 67 IT Act 2000; Sections 503, 507 IPC",
        "identity_theft": "Section 66C IT Act 2000; Section 419 IPC",
        "other": "Sections 66, 66C IT Act 2000; Relevant IPC sections as applicable",
    }
    applicable_sections = ipc_map.get(req.complaint_type.lower().replace(" ", "_"), ipc_map["other"])

    prompt = f"""You are a senior Indian police officer and legal expert. Draft a professional, legally formatted First Information Report (FIR) in English.

Complainant: {req.complainant_name}
Crime Type: {req.complaint_type}
Date of Incident: {req.incident_date}
Location: {req.incident_location}
Description: {req.description}
Accused Details: {req.accused_details or 'Unknown'}
Financial Loss: {req.loss_amount or 'Not specified'}
Applicable Legal Sections: {applicable_sections}

Generate a complete FIR with:
1. FIR Header (Station, Date, Case Number placeholder)
2. Complainant Details section
3. Incident Description (detailed, formal language)
4. Accused Details
5. Evidence/Witnesses
6. Applicable Sections of Law
7. Relief Sought
8. Declaration by complainant
9. Officer's remarks section

Use formal legal language. Include all standard FIR components. Return only the FIR text."""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=2000,
        temperature=0.3,
    )
    
    fir_text = response.choices[0].message.content
    
    # Determine IPC sections for response
    return {
        "success": True,
        "fir_text": fir_text,
        "applicable_sections": applicable_sections,
        "ai_analysis": {
            "priority": _classify_priority(req.complaint_type, req.description),
            "crime_category": req.complaint_type,
            "estimated_severity": _estimate_severity(req.description),
        }
    }


@router.post("/groq/complaint-analyze")
async def analyze_complaint(req: ComplaintAnalyzeRequest):
    """AI-powered complaint priority analysis"""
    client = get_groq_client()

    prompt = f"""Analyze this cybercrime complaint and return a JSON response:

Crime Type: {req.complaint_type}
Description: {req.description}
Evidence: {req.evidence or 'None provided'}

Return ONLY a JSON object with these exact keys:
{{
  "priority": "CRITICAL|HIGH|MEDIUM|LOW",
  "severity_score": <0-100 integer>,
  "risk_flags": ["flag1", "flag2"],
  "recommended_action": "string",
  "estimated_resolution": "string",
  "summary": "2-sentence summary"
}}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.2,
        response_format={"type": "json_object"},
    )
    
    import json
    try:
        analysis = json.loads(response.choices[0].message.content)
    except Exception:
        analysis = {
            "priority": "MEDIUM",
            "severity_score": 50,
            "risk_flags": ["Analysis pending"],
            "recommended_action": "Manual review required",
            "estimated_resolution": "3-5 business days",
            "summary": "Complaint received and queued for review."
        }
    
    return {"success": True, "analysis": analysis}


@router.post("/groq/transcribe")
async def transcribe_audio(audio: UploadFile = File(...)):
    """Transcribe audio using Groq Whisper"""
    client = get_groq_client()

    # Save temp file
    suffix = os.path.splitext(audio.filename or "audio.webm")[1] or ".webm"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        content = await audio.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        with open(tmp_path, "rb") as f:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(tmp_path), f, audio.content_type or "audio/webm"),
                model="whisper-large-v3",
                language="en",
                response_format="text",
            )
        return {"success": True, "text": transcription}
    finally:
        os.unlink(tmp_path)


@router.post("/groq/chat")
async def chat_with_ai(req: ChatRequest):
    """General AI chat for threat analysis assistance"""
    client = get_groq_client()

    system_prompt = """You are KAVACH AI Assistant — an expert in cybersecurity, deepfake detection, 
voice spoofing analysis, and phishing threat intelligence. You assist law enforcement and citizens 
in understanding digital threats. Be concise, accurate, and use technical terminology appropriately.
Always recommend official cybercrime portals (cybercrime.gov.in) for reporting."""

    messages = [{"role": "system", "content": system_prompt}]
    if req.context:
        messages.append({"role": "assistant", "content": f"Context: {req.context}"})
    messages.append({"role": "user", "content": req.message})

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        max_tokens=800,
        temperature=0.5,
    )
    
    return {
        "success": True,
        "response": response.choices[0].message.content
    }


# ── Helpers ────────────────────────────────────────────────────────────────────

def _classify_priority(crime_type: str, description: str) -> str:
    high_risk_keywords = ["deepfake", "voice", "financial", "fraud", "bank", "money", "threat", "arrest"]
    desc_lower = description.lower()
    crime_lower = crime_type.lower()
    if any(k in desc_lower or k in crime_lower for k in high_risk_keywords):
        return "HIGH"
    if len(description) > 200:
        return "MEDIUM"
    return "LOW"


def _estimate_severity(description: str) -> int:
    score = 30
    high_risk = ["financial loss", "identity", "deepfake", "nude", "threat", "blackmail", "arrest"]
    for kw in high_risk:
        if kw in description.lower():
            score += 15
    return min(score, 95)
