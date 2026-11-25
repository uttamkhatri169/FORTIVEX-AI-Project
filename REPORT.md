# Project Report: Aria, an Autonomous AI Sales Voice Agent

## 1. Objective
Build an AI agent that holds a live spoken sales conversation, qualifies the lead, answers product questions from trusted data, books demo appointments, and produces a call summary and business insights, with no human in the loop.

## 2. Solution overview
Aria is a web application. The customer talks in the browser, and a Python backend runs an LLM that can call four tools. The demo product is a fictional CRM called "Lumen CRM".

```
Browser mic -> STT -> /api/chat -> LLM <-> tools (knowledge search, availability, booking, lead update)
Browser speaker <- TTS <- reply + lead score + sentiment        SQLite CRM <-> tools
```

**Tech stack:** Python, FastAPI, Uvicorn, OpenAI-compatible LLM API (Groq, model `openai/gpt-oss-120b`; Ollama also works), SQLite, browser Web Speech API, plain HTML/JavaScript.

## 3. How each requirement is met

| Requirement | Implementation |
|---|---|
| Real-time voice (STT + TTS) | Browser Web Speech API in a listen, send, speak, listen loop |
| Contextual memory | Full message history per call, plus the last 3 call summaries per returning customer |
| Multilingual | English, Hindi and Punjabi. Aria replies in the customer's language and the TTS voice follows it |
| Lead qualification + sentiment | The `update_lead` tool runs every turn and records Budget, Authority, Need, Timeline, interest (0-10) and sentiment. Score = 15 per BANT field + 4 x interest, capped at 100 |
| RAG | The `search_knowledge` tool retrieves from `knowledge/*.md` using keyword/IDF ranking. The agent is instructed never to state prices or facts without it |
| Appointment booking + CRM | `check_availability` and `book_appointment` write to a SQLite database with double-booking protection. Lead records and calls are stored in the same database |
| Summaries + insights | On call end the LLM returns a JSON summary (summary, action items, objections, sentiment, next step). `/api/insights` reports call count, average score, booking rate and sentiment split |

## 4. Design decisions
- **Tool calling over scripted flows.** The LLM decides when to search, check slots or book, while the code stays in control of data and side effects.
- **Grounding to reduce hallucination.** Facts come only from the knowledge base, and "no match" returns a safe fallback instead of a guess.
- **Provider-agnostic.** The model name and API URL are environment variables, which allowed a quick fix when Groq retired `llama-3.3-70b-versatile` on 2026-08-16.
- **Minimal footprint.** One backend file and one frontend file, with no build step, which keeps the project easy to read and run.

## 5. Limitations (honest assessment)
- Voice is turn-based, with no streaming or barge-in, and works only in Chrome or Edge. Punjabi voice availability varies by device.
- RAG is keyword-based, not embedding-based.
- The CRM is a local SQLite file, not an external system such as HubSpot, and no confirmation email or calendar invite is sent.
- Sessions are held in memory, and only calls ended with **End call** are saved.
- No authentication or rate limiting, and lead fields are rendered with `innerHTML`, which should be changed to `textContent` before public deployment.
- The system has not been evaluated with a formal test set, so accuracy and latency figures are not reported.

## 6. Future work
1. Streaming STT/TTS (Whisper or Deepgram with ElevenLabs) for low-latency, interruptible calls.
2. Embedding-based retrieval with a vector database (Qdrant or pgvector).
3. Real CRM integration (HubSpot or Zoho) and email, WhatsApp or calendar confirmations.
4. Automatic language detection.
5. An evaluation suite of scripted calls covering booking, scoring and multilingual behaviour.
6. Authentication, persistent sessions and logging for production use.

## 7. Repository contents
`app.py` (backend), `static/index.html` (UI and voice logic), `knowledge/` (RAG source), `requirements.txt`, `.env.example`, `README.md` (quick start), `DOCUMENTATION.md` (full code explanation) and this report.
