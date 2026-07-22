# 🛡️ NovaShield — AI-Powered Digital Public Safety Platform

NovaShield is an AI-powered cybercrime detection and reporting platform built for India's digital safety ecosystem. It detects deepfakes, voice spoofing, and phishing URLs, and lets citizens file AI-assisted FIR drafts and complaint reports — all in a single web app.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎭 **Deepfake Detection** | EfficientNet-B0 + Vision Transformer analyzes video/image for facial manipulation |
| 🎤 **Voice Spoofing Detection** | 89 spectral features (MFCC, Chroma, ZCR) detect TTS synthesis & replay attacks |
| 🔗 **Phishing URL Scanner** | XGBoost + 25 heuristic rules detect malicious URLs in real-time |
| 📝 **AI Complaint Portal** | NLP-based severity scoring, priority auto-tagging, timeline tracking |
| 📄 **FIR Auto-Draft** | Groq Llama 3.3 70B generates Section 65B-compliant legal FIR drafts |
| 🎙️ **Voice I/O** | Speak inputs via Groq Whisper STT; AI reads results via Browser TTS |
| 🤖 **Legal Chat Assistant** | Ask about IPC sections, cybercrime laws, and rights |
| 📊 **Analytics Dashboard** | Threat trends, prediction charts, priority breakdowns |
| ⚙️ **Admin Panel** | User management, role control, audit logs, system health |

---

## 🏗️ Tech Stack

- **Backend:** FastAPI + SQLAlchemy (async) + Alembic
- **Database:** PostgreSQL (SQLite for local dev)
- **AI/ML:** PyTorch, XGBoost, Librosa, Groq (Whisper + Llama 3.3 70B)
- **Auth:** JWT (HttpOnly cookies) + Argon2id hashing
- **Frontend:** Jinja2 templates, Chart.js, vanilla JS
- **Storage:** Local filesystem (S3-compatible optional)

---

## 🚀 Deploying on Render

### Files to push to GitHub

Push the **entire repository** — all of the following directories and files are required:

```
backend/
├── app/
│   ├── main.py
│   ├── api/
│   ├── core/
│   ├── db/
│   ├── models/
│   ├── schemas/
│   ├── services/
│   ├── utils/
│   └── worker.py
├── alembic/
├── alembic.ini
├── templates/          ← Jinja2 HTML templates
├── static/             ← CSS and JS assets
├── requirements.txt
└── .env.example        ← rename to .env for local dev

README.md
```

> **Do NOT push:** `.env`, `uploads/`, `models/pretrained/*.pth`, `novashield.db`, `__pycache__/`

Add a `.gitignore`:
```gitignore
.env
uploads/
*.db
*.pth
*.pkl
__pycache__/
.pythonlibs/
*.egg-info/
.local/
```

---

### Step-by-Step Render Deployment

#### 1. Create a PostgreSQL database on Render

1. Go to [render.com](https://render.com) → **New → PostgreSQL**
2. Name it `novashield-db`, choose the free tier
3. Copy the **Internal Database URL** (format: `postgresql://...`)

#### 2. Create a Web Service on Render

1. Go to **New → Web Service**
2. Connect your GitHub repository
3. Configure:

| Setting | Value |
|---------|-------|
| **Name** | `novashield` |
| **Root Directory** | `backend` |
| **Runtime** | Python 3 |
| **Build Command** | `pip install -r requirements.txt` |
| **Start Command** | `uvicorn app.main:app --host 0.0.0.0 --port $PORT` |

#### 3. Set Environment Variables

In the Render dashboard → **Environment**, add:

| Variable | Value | Required |
|----------|-------|----------|
| `DATABASE_URL` | Your Render PostgreSQL Internal URL | ✅ |
| `SECRET_KEY` | A random 64-char string | ✅ |
| `GROQ_API_KEY` | Your [Groq API key](https://console.groq.com) | ✅ |
| `ENVIRONMENT` | `production` | ✅ |
| `SESSION_SECRET` | Another random 64-char string | ✅ |
| `REDIS_URL` | Optional Redis URL (Upstash free tier) | ❌ |
| `SMTP_HOST` | Your SMTP host (for email notifications) | ❌ |
| `SMTP_USER` | SMTP username | ❌ |
| `SMTP_PASSWORD` | SMTP password | ❌ |

Generate a secure `SECRET_KEY`:
```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

#### 4. Run Database Migrations (optional)

If you want to use Alembic migrations instead of auto-create:

In Render → **Shell** tab, run:
```bash
cd backend && alembic upgrade head
```

Or add it to your build command:
```
pip install -r requirements.txt && alembic upgrade head
```

#### 5. Deploy

Click **Deploy** — Render will build and start your service. The first deploy creates database tables automatically via SQLAlchemy's `Base.metadata.create_all`.

---

## 🔑 Default Credentials

After first deploy, a default admin account is auto-created:

| Field | Value |
|-------|-------|
| Email | `admin@novashield.ai` |
| Password | `Admin@123` |

**Change this password immediately after first login.**

---

## 📋 API Reference

Interactive docs available at:
- **Swagger UI:** `https://your-app.onrender.com/api/docs`
- **ReDoc:** `https://your-app.onrender.com/api/redoc`

### Key Endpoints

```
POST /api/v1/auth/register       Register new user
POST /api/v1/auth/login          Login (sets HttpOnly cookies)
POST /api/v1/auth/logout         Logout
GET  /api/v1/auth/me             Get current user profile

POST /api/v1/detect/deepfake     Upload video for deepfake analysis
POST /api/v1/detect/voice        Upload audio for voice spoofing analysis
POST /api/v1/detect/phishing     Analyze URL for phishing
GET  /api/v1/detections/{id}     Poll detection result
GET  /api/v1/detections/{id}/report  Download Section 65B report
GET  /api/v1/history             Get scan history (paginated)
GET  /api/v1/stats               Get user scan statistics

POST /api/v1/complaints/submit   Submit cybercrime complaint
GET  /api/v1/complaints/track/{id}  Track complaint status
GET  /api/v1/complaints/all      List all complaints (yours / all for admin)

POST /api/v1/groq/fir-draft      Generate FIR with Groq AI
POST /api/v1/groq/chat           Chat with legal AI assistant
POST /api/v1/groq/transcribe     Transcribe audio via Groq Whisper

GET  /api/v1/admin/users         List all users (admin only)
PUT  /api/v1/admin/users/{id}/role  Update user role (admin only)
GET  /api/v1/admin/audit-logs    View audit logs (admin only)
GET  /api/v1/admin/stats         System-wide statistics (admin only)

GET  /health                     Health check
GET  /ready                      Readiness check (DB connectivity)
GET  /metrics                    Basic platform metrics
```

---

## 🛠️ Local Development

```bash
# 1. Clone the repo
git clone https://github.com/your-username/novashield.git
cd novashield/backend

# 2. Install dependencies
pip install -r requirements.txt

# 3. Create .env file
cp .env.example .env
# Edit .env — add your GROQ_API_KEY at minimum

# 4. Start the server
uvicorn app.main:app --host 0.0.0.0 --port 5000 --reload

# 5. Open in browser
# http://localhost:5000
```

The app uses SQLite by default for local development — no database setup needed.

---

## 🔒 Security Notes

- Passwords hashed with **Argon2id** (memory-hard, secure)
- Auth tokens stored in **HttpOnly Secure cookies** (XSS-safe)
- All user inputs HTML-escaped to prevent injection
- File uploads validated by magic bytes (not just extension)
- Rate limiting: 20 requests/hour per IP
- Audit logging for all auth events

---

## 📄 License

MIT License — free to use, modify, and distribute.

Built with ❤️ for the ET AI Hackathon · Digital India Initiative · © 2026 NovaShield
