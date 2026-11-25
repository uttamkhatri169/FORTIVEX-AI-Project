import glob, json, math, os, re, sqlite3
from collections import Counter
from openai import OpenAI
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# llama-3.3-70b-versatile was shut down on Groq on 2026-08-16. Override with MODEL=... if Groq retires this one too.
load_dotenv()  # reads GROQ_API_KEY / MODEL from a .env file next to app.py
MODEL = os.getenv("MODEL", "openai/gpt-oss-120b")
# gpt-oss models "think" before answering; keep that short so replies stay fast and token budgets are not eaten.
EXTRA = {"reasoning_effort": "low"} if "gpt-oss" in MODEL else {}
# Free LLM via any OpenAI-compatible endpoint (default: Groq free tier). Ollama works too: LLM_BASE_URL=http://localhost:11434/v1
API_KEY = os.getenv("GROQ_API_KEY", "").strip().strip("\"'")
if not API_KEY or "your" in API_KEY.lower():
    print("\n*** WARNING: GROQ_API_KEY is missing or still the placeholder. Create a .env file with GROQ_API_KEY=gsk_... ***\n")
elif not API_KEY.startswith("gsk_") and "groq.com" in os.getenv("LLM_BASE_URL", "groq.com"):
    print("\n*** WARNING: Groq keys start with 'gsk_'. This key looks like it is from another provider. ***\n")
client = OpenAI(api_key=API_KEY or "none", base_url=os.getenv("LLM_BASE_URL", "https://api.groq.com/openai/v1"))
db = sqlite3.connect("crm.db", check_same_thread=False)
db.executescript("""
create table if not exists calls(id integer primary key autoincrement, lead_key text, lead text, summary text, score int, sentiment text, booked int);
create table if not exists appointments(id integer primary key autoincrement, day text, slot text, name text, email text);""")

# ---------- RAG (keyword/IDF retrieval; swap for embeddings + a vector DB when you scale) ----------
tok = lambda s: set(re.findall(r"\w+", s.lower()))
CHUNKS = [p.strip() for f in glob.glob("knowledge/*.md") for p in open(f, encoding="utf8").read().split("\n\n") if p.strip()]
DF = Counter(w for c in CHUNKS for w in tok(c))

def search(q, k=3):
    qs = tok(q)
    ranked = sorted(CHUNKS, key=lambda c: -sum(math.log(1 + len(CHUNKS) / DF[w]) for w in qs & tok(c)))
    return [c for c in ranked[:k] if qs & tok(c)]

# ---------- Tools the agent can call ----------
SLOTS = ["10:00", "11:00", "12:00", "14:00", "15:00", "16:00"]
TOOLS = [
    {"name": "search_knowledge", "description": "Look up product, pricing, security or support facts. Use before answering any factual question.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "check_availability", "description": "List free demo slots for a date (YYYY-MM-DD).",
     "input_schema": {"type": "object", "properties": {"date": {"type": "string"}}, "required": ["date"]}},
    {"name": "book_appointment", "description": "Book a demo slot after the customer confirmed it aloud.",
     "input_schema": {"type": "object", "properties": {"date": {"type": "string"}, "time": {"type": "string"}, "name": {"type": "string"}, "email": {"type": "string"}}, "required": ["date", "time", "name"]}},
    {"name": "update_lead", "description": "Call EVERY turn: save qualification facts learned so far, the customer's current sentiment and their language.",
     "input_schema": {"type": "object", "properties": {
         "name": {"type": "string"}, "company": {"type": "string"}, "budget": {"type": "string"}, "authority": {"type": "string"},
         "need": {"type": "string"}, "timeline": {"type": "string"}, "interest": {"type": "integer", "description": "0-10"},
         "sentiment": {"type": "string", "enum": ["positive", "neutral", "negative"]},
         "language": {"type": "string", "description": "BCP-47 code of the customer's language: en-IN, hi-IN or pa-IN"}}, "required": ["sentiment", "language"]}},
]

TOOLS_OAI = [{"type": "function", "function": {"name": t["name"], "description": t["description"], "parameters": t["input_schema"]}} for t in TOOLS]

def run_tool(name, a, s):
    if name == "search_knowledge":
        return "\n---\n".join(search(a["query"])) or "No match. Say you will check with the team."
    if name == "check_availability":
        taken = {r[0] for r in db.execute("select slot from appointments where day=?", (a["date"],))}
        return json.dumps([x for x in SLOTS if x not in taken])
    if name == "book_appointment":
        if db.execute("select 1 from appointments where day=? and slot=?", (a["date"], a["time"])).fetchone():
            return "Slot already taken. Offer another."
        db.execute("insert into appointments(day,slot,name,email) values(?,?,?,?)", (a["date"], a["time"], a["name"], a.get("email", "")))
        db.commit(); s["booked"] = True
        return "Booked."
    s["lead"].update({k: v for k, v in a.items() if v not in (None, "")})  # update_lead (the CRM record)
    return "Saved."

def score(l):  # BANT completeness (60) + interest (40)
    return min(100, 15 * sum(bool(l.get(k)) for k in ("budget", "authority", "need", "timeline")) + 4 * int(l.get("interest", 0)))

def system_prompt(key):
    past = [r[0] for r in db.execute("select summary from calls where lead_key=? order by id desc limit 3", (key,))]
    mem = ("\nEarlier calls with this customer (use naturally, do not recite): " + " | ".join(past)) if past else ""
    return ("You are Aria, a friendly outbound sales agent for Lumen CRM, speaking on a live phone call. "
            "Keep replies to 1-3 short spoken sentences, no lists or markdown. Reply in the language the customer speaks (English, Hindi or Punjabi). "
            "Goals: discover budget, authority, need and timeline naturally (one question at a time), answer using search_knowledge only (never invent prices), "
            "handle objections empathetically, and offer a demo; confirm date, time, name and email aloud before book_appointment. "
            "Call update_lead every turn. Be honest that you are an AI if asked; respect 'not interested'." + mem)

# ---------- API ----------
app = FastAPI()
SESSIONS = {}

class Chat(BaseModel):
    session_id: str
    lead_key: str = "anon"
    text: str

@app.post("/api/chat")
def chat(c: Chat):
    s = SESSIONS.setdefault(c.session_id, {"msgs": [], "lead": {}, "booked": False, "key": c.lead_key})
    msgs, events = s["msgs"], []
    msgs.append({"role": "user", "content": c.text})
    try:
        for _ in range(6):
            r = client.chat.completions.create(model=MODEL, max_tokens=1500, tools=TOOLS_OAI, extra_body=EXTRA,
                                               messages=[{"role": "system", "content": system_prompt(s["key"])}] + msgs)
            m = r.choices[0].message
            entry = {"role": "assistant", "content": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = [{"id": t.id, "type": "function", "function": {"name": t.function.name, "arguments": t.function.arguments}} for t in m.tool_calls]
            msgs.append(entry)
            if not m.tool_calls:
                break
            for t in m.tool_calls:
                events.append(t.function.name)
                try:
                    out = run_tool(t.function.name, json.loads(t.function.arguments or "{}"), s)
                except Exception as e:  # small free models sometimes send bad arguments
                    out = f"Tool error: {e}. Try again with valid arguments."
                msgs.append({"role": "tool", "tool_call_id": t.id, "content": out})
    except Exception as e:
        raise HTTPException(502, f"LLM error: {e}")
    return {"reply": m.content or "", "lead": s["lead"], "score": score(s["lead"]), "events": events}

def transcript(s):
    return "\n".join(("Customer: " if m["role"] == "user" else "Agent: ") + m["content"]
                     for m in s["msgs"] if m["role"] in ("user", "assistant") and m.get("content"))

@app.post("/api/end/{sid}")
def end(sid: str):
    s = SESSIONS.pop(sid, None)
    if not s:
        raise HTTPException(404, "Unknown session")
    r = client.chat.completions.create(model=MODEL, max_tokens=2000, extra_body=EXTRA, messages=[{"role": "user", "content":
        "Summarise this sales call. Return ONLY JSON with keys: summary (string), action_items (list), objections (list), "
        "sentiment (positive|neutral|negative), next_step (string).\n\n" + transcript(s)}])
    raw = re.sub(r"```(?:json)?", "", r.choices[0].message.content or "").strip()
    try:
        out = json.loads(raw)
    except ValueError:
        out = {"summary": raw, "action_items": [], "objections": [], "sentiment": "neutral", "next_step": ""}
    db.execute("insert into calls(lead_key,lead,summary,score,sentiment,booked) values(?,?,?,?,?,?)",
               (s["key"], json.dumps(s["lead"]), out.get("summary", ""), score(s["lead"]), out.get("sentiment", "neutral"), int(s["booked"])))
    db.commit()
    return {**out, "lead": s["lead"], "score": score(s["lead"]), "booked": s["booked"]}

@app.get("/api/insights")
def insights():
    rows = db.execute("select score, sentiment, booked, summary from calls").fetchall()
    n = len(rows) or 1
    return {"calls": len(rows), "avg_score": round(sum(r[0] for r in rows) / n),
            "booking_rate": round(100 * sum(r[2] for r in rows) / n),
            "sentiment": dict(Counter(r[1] for r in rows))}

@app.get("/")
def home():
    return FileResponse("static/index.html")
app.mount("/static", StaticFiles(directory="static"))