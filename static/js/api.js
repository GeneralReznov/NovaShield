/**
 * NovaShield — Shared API Utilities + Toast + Voice I/O
 * v3 — Futuristic Edition
 */

// ── Toast Container ──────────────────────────────────────────────────────────
const toastContainer = document.createElement('div');
toastContainer.className = 'toast-container';
document.body.appendChild(toastContainer);

function showToast(message, type = 'info', duration = 4000) {
  const icons = { success: '✅', error: '❌', info: '🔷', warning: '⚠️' };
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <span style="font-size:18px;flex-shrink:0">${icons[type] || icons.info}</span>
    <span style="flex:1;font-size:13px;line-height:1.4">${escapeHtml(String(message))}</span>
    <button onclick="this.parentElement.remove()" style="background:none;border:none;color:var(--text-muted);cursor:pointer;font-size:18px;flex-shrink:0;padding:0 0 0 4px">✕</button>`;
  toastContainer.appendChild(toast);
  const t = setTimeout(() => {
    if (toast.parentElement) {
      toast.classList.add('removing');
      setTimeout(() => toast.remove(), 320);
    }
  }, duration);
  toast.querySelector('button').addEventListener('click', () => clearTimeout(t));
  return toast;
}

// ── API Request Helper ────────────────────────────────────────────────────────
async function apiRequest(method, url, body = null, isForm = false) {
  const opts = { method, credentials: 'include', headers: {} };
  if (body && !isForm) {
    opts.headers['Content-Type'] = 'application/json';
    opts.body = JSON.stringify(body);
  } else if (body && isForm) {
    opts.body = body;
  }
  const res = await fetch(url, opts);
  if (res.status === 401) {
    window.location.href = '/login';
    throw new Error('Session expired. Please log in.');
  }
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    const msg = data?.detail?.message || data?.detail || data?.message || `HTTP ${res.status}`;
    throw new Error(typeof msg === 'string' ? msg : JSON.stringify(msg));
  }
  return data;
}

const api = {
  get:    (url)        => apiRequest('GET', url),
  post:   (url, body)  => apiRequest('POST', url, body),
  put:    (url, body)  => apiRequest('PUT', url, body),
  delete: (url)        => apiRequest('DELETE', url),
  upload: (url, form)  => apiRequest('POST', url, form, true),
};

// ── Auth Helpers ──────────────────────────────────────────────────────────────
async function requireAuth() {
  try {
    const user = await api.get('/api/v1/auth/me');
    return user;
  } catch {
    window.location.href = '/login';
    return null;
  }
}

async function getUser() {
  try { return await api.get('/api/v1/auth/me'); } catch { return null; }
}

function logout() {
  api.post('/api/v1/auth/logout', {}).finally(() => { window.location.href = '/login'; });
}

// ── Format Helpers ────────────────────────────────────────────────────────────
function formatDate(str) {
  if (!str) return '—';
  try {
    return new Date(str).toLocaleString('en-IN', { dateStyle: 'medium', timeStyle: 'short' });
  } catch { return str; }
}
function capitalize(s) { return s ? s.charAt(0).toUpperCase() + s.slice(1).toLowerCase() : ''; }
function escapeHtml(str) {
  return String(str)
    .replace(/&/g,'&amp;').replace(/</g,'&lt;')
    .replace(/>/g,'&gt;').replace(/"/g,'&quot;').replace(/'/g,'&#39;');
}
function animateCounter(el, target, duration = 1500) {
  const step = target / (duration / 16);
  let current = 0;
  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = Math.floor(current).toLocaleString();
    if (current >= target) clearInterval(timer);
  }, 16);
}

// ── Voice Input (STT via Groq Whisper) ───────────────────────────────────────
class VoiceInput {
  constructor(onTranscript, onStateChange) {
    this.onTranscript = onTranscript;
    this.onStateChange = onStateChange;
    this.mediaRecorder = null;
    this.chunks = [];
    this.recording = false;
    this.stream = null;
  }

  async toggle() {
    this.recording ? this.stop() : await this.start();
  }

  async start() {
    if (this.recording) return;
    try {
      this.stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: true, noiseSuppression: true } });
      const mimeType = ['audio/webm;codecs=opus','audio/webm','audio/ogg;codecs=opus','audio/mp4']
        .find(t => MediaRecorder.isTypeSupported(t)) || '';
      this.mediaRecorder = new MediaRecorder(this.stream, mimeType ? { mimeType } : {});
      this.chunks = [];
      this.mediaRecorder.ondataavailable = e => { if (e.data && e.data.size > 0) this.chunks.push(e.data); };
      this.mediaRecorder.onstop = () => this._sendAudio();
      this.mediaRecorder.start(250); // collect data every 250ms
      this.recording = true;
      this.onStateChange?.('recording');
    } catch (err) {
      const msg = err.name === 'NotAllowedError' ? 'Microphone access denied. Please allow microphone access.' : `Mic error: ${err.message}`;
      showToast(msg, 'error');
      this.onStateChange?.('idle');
    }
  }

  stop() {
    if (this.mediaRecorder && this.recording) {
      this.mediaRecorder.stop();
      this.recording = false;
      this.onStateChange?.('processing');
    }
  }

  async _sendAudio() {
    if (this.stream) { this.stream.getTracks().forEach(t => t.stop()); this.stream = null; }
    if (!this.chunks.length) { this.onStateChange?.('idle'); return; }

    const mimeType = this.mediaRecorder?.mimeType || 'audio/webm';
    const ext = mimeType.includes('mp4') ? '.mp4' : mimeType.includes('ogg') ? '.ogg' : '.webm';
    const blob = new Blob(this.chunks, { type: mimeType });
    const form = new FormData();
    form.append('audio', blob, `recording${ext}`);

    try {
      const res = await api.upload('/api/v1/groq/transcribe', form);
      if (res.success && res.text && res.text.trim()) {
        this.onTranscript?.(res.text.trim());
      } else {
        showToast('No speech detected. Please try again.', 'warning');
      }
    } catch (err) {
      showToast('Transcription failed: ' + err.message, 'error');
    } finally {
      this.onStateChange?.('idle');
    }
  }
}

// ── Text-to-Speech (Browser Native) ──────────────────────────────────────────
class VoiceOutput {
  constructor() {
    this.synth = window.speechSynthesis;
    this.speaking = false;
    this._utterance = null;
  }

  speak(text, onStart, onEnd) {
    if (!this.synth) { showToast('Text-to-speech not supported in this browser.', 'warning'); return; }
    this.stop();
    const clean = text.replace(/[#*_`]/g, '').replace(/\s+/g, ' ').trim();
    const utter = new SpeechSynthesisUtterance(clean);
    utter.rate = 0.95;
    utter.pitch = 1.05;
    utter.volume = 1.0;
    utter.lang = 'en-US';

    // Pick best voice
    const trySetVoice = () => {
      const voices = this.synth.getVoices();
      const pick = voices.find(v => v.name.includes('Google') && v.lang === 'en-US')
        || voices.find(v => v.name.includes('Microsoft') && v.lang === 'en-US')
        || voices.find(v => v.lang === 'en-US' && !v.localService)
        || voices.find(v => v.lang.startsWith('en'));
      if (pick) utter.voice = pick;
    };

    if (this.synth.getVoices().length) trySetVoice();
    else this.synth.onvoiceschanged = trySetVoice;

    utter.onstart = () => { this.speaking = true; onStart?.(); };
    utter.onend = () => { this.speaking = false; this._utterance = null; onEnd?.(); };
    utter.onerror = () => { this.speaking = false; this._utterance = null; onEnd?.(); };
    this._utterance = utter;
    this.synth.speak(utter);
  }

  stop() {
    if (this.synth) { this.synth.cancel(); }
    this.speaking = false;
    this._utterance = null;
  }
}

// ── Drag-and-Drop Dropzone ────────────────────────────────────────────────────
function initDropzone(dropEl, inputEl, onFile) {
  dropEl.addEventListener('click', () => inputEl.click());
  dropEl.addEventListener('dragover', e => { e.preventDefault(); dropEl.classList.add('drag-over'); });
  dropEl.addEventListener('dragleave', e => { if (!dropEl.contains(e.relatedTarget)) dropEl.classList.remove('drag-over'); });
  dropEl.addEventListener('drop', e => {
    e.preventDefault(); dropEl.classList.remove('drag-over');
    const file = e.dataTransfer.files[0];
    if (file) onFile(file);
  });
  inputEl.addEventListener('change', e => { if (e.target.files[0]) onFile(e.target.files[0]); });
}

// ── Polling Helper ─────────────────────────────────────────────────────────────
async function pollResult(detectionId, onResult, maxAttempts = 40) {
  let attempts = 0;
  const interval = setInterval(async () => {
    attempts++;
    try {
      const data = await api.get(`/api/v1/detections/${detectionId}`);
      if (data.status === 'completed' || data.status === 'failed') {
        clearInterval(interval);
        onResult(data);
      } else if (attempts >= maxAttempts) {
        clearInterval(interval);
        onResult({ status: 'failed', error: 'Analysis timed out. Please try again.', confidence: 0, is_fake: false });
      }
    } catch {
      if (attempts >= maxAttempts) {
        clearInterval(interval);
        onResult({ status: 'failed', error: 'Connection lost. Please refresh.', confidence: 0, is_fake: false });
      }
    }
  }, 2500);
}

// ── Populate User Nav ──────────────────────────────────────────────────────────
async function populateUserNav() {
  const user = await getUser();
  if (!user) return;
  const initial = (user.username || user.email || '?')[0].toUpperCase();
  document.querySelectorAll('.user-avatar').forEach(el => el.textContent = initial);
  document.querySelectorAll('.user-name').forEach(el => el.textContent = user.username || user.email);
  if (user.role === 'admin') {
    document.querySelectorAll('.admin-only').forEach(el => el.classList.remove('hidden'));
  }
  return user;
}
