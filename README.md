# Aria: Autonomous AI Sales Voice Agent (Python + browser demo)

## Run
    python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
    pip install -r requirements.txt
    export GROQ_API_KEY=your-free-key                  # Windows: set GROQ_API_KEY=your-key
    uvicorn app:app --reload
Optional: `export MODEL=openai/gpt-oss-120b` (default). If you ever see `model_not_found`, Groq has retired the model; pick a current one from https://console.groq.com/docs/models.
Open http://localhost:8000 in Chrome or Edge, click **Start call**, allow the mic.

## Requirement map
| Requirement | Where |
|---|---|
| Real-time STT + TTS | Browser Web Speech API (`static/index.html`): listen, send, speak, listen loop |
| Contextual memory | In-session message history, plus past call summaries per browser (`system_prompt`) |
| Multilingual | English, Hindi, Punjabi: agent replies in the caller's language, TTS voice follows `language` |
| Lead qualification + sentiment | `update_lead` tool each turn, BANT + interest score, sentiment dots |
| RAG | `knowledge/*.md`, `search()`; the agent must call `search_knowledge` before facts |
| Booking + CRM | `check_availability`, `book_appointment`, SQLite `crm.db` (calls, appointments) |
| Summaries + insights | `/api/end/{id}` produces JSON summary; `/api/insights` aggregates |

## Next steps
Replace keyword RAG with embeddings (pgvector/Qdrant); swap Web Speech for Deepgram + ElevenLabs over WebSockets for lower latency; push leads to HubSpot; add an eval set of scripted calls.
