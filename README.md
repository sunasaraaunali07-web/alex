# Alex — Voice Partner

**Alex** is a voice-first AI English conversation partner designed to help learners speak naturally and fluently. Speak freely via your microphone, receive natural spoken replies paired with gentle, pinpointed grammar coaching, and watch an audio-reactive particle globe respond in real time.

---

## Highlights

- **Reactive 3D Particle Globe UI** — A 950-point Fibonacci-sphere HTML5 canvas (`app.js`) that breathes with live microphone frequencies while you speak and smoothly docks (`270px` → `122px`) once your conversation starts, giving maximum room to the chat feed.
- **3-Zone Zero-Overlap Layout (`index.html`)** — Top pinned header & globe, smooth scrollable chat region with a `22px` top fade mask, and a frosted-glass bottom control dock in normal flex flow (`100dvh` edge-to-edge on mobile with `visualViewport` keyboard support; centered phone card on desktop).
- **Speech-Synced Highlights & Inline Tips** — Whisper transcription (`whisper-large-v3-turbo`) highlights exact phrasing errors directly inside your chat bubble (`<mark>`). Inline **Alex’s English Tips** cards show concise *Instead of / Say / Why* diffs and let you replay individual tip audio (**▶ Tip audio**).
- **Active-Segment Auto-Scroll** — Shows your spoken transcript first, smoothly unfolds corrections below it, and tracks along with Alex’s voice across Tip 1, Tip 2, and Alex’s reply without jumping past what you just said.
- **Segmented Edge TTS & Adaptive Pauses** — `pause_manager.py` and `tts_service.py` synthesize separate audio clips for top corrections and the conversational reply in parallel, inserting complexity-calibrated breathing pauses between tips (`P = B + C·K + M·D`). Includes browser `speechSynthesis` fallback.
- **Persistent Learning Memory Vault** — Asynchronous SQLite memory extraction (`memory_service.py`) builds a persistent learner profile (background, goals, key facts, struggled concepts, and practiced topics) across sessions with zero turn-latency impact.

---

## Architecture Overview

```text
ai-english-friend/
├── backend/
│   ├── app/
│   │   ├── main.py                     # FastAPI server, API routes, static frontend mount
│   │   ├── core/config.py              # Live .env configuration via pydantic-settings
│   │   └── services/
│   │       ├── stt_service.py          # Groq Whisper STT + acoustic confidence metadata
│   │       ├── llm_service.py          # Grammar audit + conversational response (Groq / Gemini)
│   │       ├── tts_service.py          # Parallel Edge TTS segment synthesis
│   │       ├── pause_manager.py        # Adaptive pause timing between corrections & reply
│   │       ├── memory_service.py       # SQLite persistent Learning Memory Vault
│   │       ├── session_manager.py      # In-memory multi-turn dialogue window
│   │       └── analysis_service.py     # Post-session linguistic analysis utilities
├── frontend/
│   ├── index.html                      # 3-zone mobile/desktop UI layout & styling
│   └── app.js                          # Mic capture, Web Audio globe visualizer, speech-synced chat
├── .env.example                        # Environment variable template
├── requirements.txt                    # Python dependencies
└── start.bat                           # One-click Windows launcher
```

---

## Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/sunasaraaunali07-web/alex.git
cd alex
```

### 2. Create a Python virtual environment & install dependencies
```bash
python -m venv .venv
```

**Windows (PowerShell):**
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 3. Configure environment variables
Copy `.env.example` to `.env` and add your API key(s):
```bash
cp .env.example .env
```

```env
GROQ_API_KEY=your_groq_api_key_here
GEMINI_API_KEY=your_gemini_api_key_here   # optional fallback / model option
```

### 4. Start the app
**Windows:** double-click `start.bat`, or run manually:
```powershell
$env:PYTHONPATH="backend"
.\.venv\Scripts\python.exe -m uvicorn app.main:app --host 127.0.0.1 --port 8000
```

**macOS / Linux:**
```bash
PYTHONPATH=backend uvicorn app.main:app --host 127.0.0.1 --port 8000
```

Then open **http://localhost:8000/** in your browser and allow microphone access.

---

## Configuration Reference

| Variable | Default | Purpose |
| :--- | :--- | :--- |
| `GROQ_API_KEY` | `""` | API key for Whisper STT and Groq LLM inference |
| `GEMINI_API_KEY` | `""` | Optional Google Gemini API key |
| `GROQ_STT_MODEL` | `whisper-large-v3-turbo` | Speech-to-text model on Groq |
| `GROQ_LLM_MODEL` | `qwen/qwen3.8-27b` | Primary conversational & grammar coaching model |
| `GEMINI_MODEL` | `gemini-3.8-flash` | Gemini model option |
| `DEFAULT_TTS_VOICE` | `en-US-JennyNeural` | Primary Edge TTS neural voice |
| `FALLBACK_TTS_VOICE` | `en-US-GuyNeural` | Alternate Edge TTS voice |

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/api/health` | Verifies API status and whether Groq/Gemini keys are configured |
| `POST` | `/api/turn` | Accepts `audio` (`multipart/form-data`), `session_id`, and `user_id`; returns transcript, deduplicated corrections, reply text, segmented base64 audio, and latency metrics |
| `GET` | `/api/user/{user_id}/memory` | Retrieves the user's persistent SQLite Learning Memory Vault profile and history |

---

## Controls & Usage Tips

- **Tap to speak** → Start speaking naturally; tap **Tap when you’re done** when finished.
- **Tap to interrupt** → Interrupt Alex anytime while he is speaking.
- **Tap underlined words (`mark`) or `✎ N corrections ▾`** → Expand or collapse inline grammar & phrasing tips.
- **▶ Tip audio / Tap Alex’s bubble** → Replay the audio for an individual tip or Alex’s response.
- **✦ button (bottom-left)** → Toggle state previews and click **↻ New topic** to start a fresh conversation thread while keeping your long-term Learning Memory Vault.
