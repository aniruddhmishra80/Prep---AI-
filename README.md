# AI Voice Interview Platform

A voice-driven AI interviewer. It reads your resume and a job description, works out
the gap between them, asks adaptive questions grounded in *your* background, scores
every answer on three independent signals, and writes you a report.

Built with FastAPI, LangChain, ChromaDB and Streamlit.

---

## What it does

1. You upload a resume PDF and a job description.
2. An LLM extracts the skills from both, and set operations compute the gap:
   matched, missing, extra, and a match ratio.
3. Both documents are chunked, embedded and stored in a vector store scoped to
   your session.
4. For each question the system retrieves the chunks most relevant to the target
   skill and asks the LLM for a question at the current difficulty level.
5. Your answer gets three scores — keyword coverage, semantic similarity to a model
   answer, and an LLM rubric grade — blended into one number.
6. That number moves the next question's difficulty up or down.
7. After 8 questions you get a report: overall score, per-skill breakdown,
   strengths, weaknesses and concrete tips.

In voice mode the question is read aloud with gTTS, your spoken answer is
transcribed locally by faster-whisper, and a speaking analyser scores your delivery
(filler-word rate, answer length) separately from your content.

---

## Setup

Five commands from a clean machine.

```bash
# 1. get the code and enter it
cd ai-interview-platform

# 2. virtual environment
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# 3. dependencies
pip install -r requirements.txt

# 4. configuration
cp .env.example .env              # Windows: copy .env.example .env
#    then open .env and paste ONE api key

# 5. run it — two terminals, both from the project root
uvicorn backend.main:app --reload     # terminal 1 -> http://127.0.0.1:8000/docs
streamlit run frontend/app.py         # terminal 2 -> http://localhost:8501
```

### Getting an API key

You only need one. All three have a free tier.

| Provider | `LLM_PROVIDER` | Where |
|---|---|---|
| Google Gemini (default) | `google` | https://aistudio.google.com/app/apikey |
| Mistral | `mistral` | https://console.mistral.ai |
| Ollama (local, no key at all) | `ollama` | `ollama pull llama3.1` |

Embeddings default to `local`, which downloads `all-MiniLM-L6-v2` (~80MB) on first
run and then needs no internet and costs nothing.

**Fully offline mode:** set `LLM_PROVIDER=ollama` and `EMBEDDING_PROVIDER=local`.
No key, no internet, no rate limits. Useful for demos on bad conference wifi.

### If voice fails to install

`PyAudio` needs system audio libraries and often fails on Windows. Voice is
optional — delete the voice block at the bottom of `requirements.txt` and everything
else works in text mode. Both halves also fall back on their own: if gTTS can't
reach the network it uses `pyttsx3`, and if Whisper isn't installed it uses
SpeechRecognition's Google endpoint.

```bash
# Ubuntu
sudo apt install portaudio19-dev python3-pyaudio espeak
# macOS
brew install portaudio
```

### Check it works before touching the UI

```bash
python test_pipeline.py       # offline logic tests, no API key needed
```

Then open http://127.0.0.1:8000/docs and try each endpoint from Swagger.

---

## Architecture

```
Streamlit  (frontend/app.py — thin client, zero business logic)
    │  HTTP via requests
    ▼
FastAPI  (backend/main.py)
    │
    ├── routers/     request and response translation only
    │
    └── core/        all the logic lives here
         ├── parser.py      PDF → text            (PyPDFLoader)
         ├── skills.py      text → skill list     (LLM + PydanticOutputParser)
         ├── gap.py         two lists → the gap   (set operations)
         ├── rag.py         split, embed, store, retrieve  (Chroma)
         ├── questions.py   skill + context → question     (LLM)
         ├── scoring.py     answer → three scores → one    (keyword/cosine/LLM)
         ├── interview.py   score → next difficulty and skill
         └── report.py       transcript → written feedback  (LLM)
    │
    ▼
SQLite (sessions and answers)  +  Chroma (vectors)  +  Gemini / Mistral / Ollama
```

### Folder layout

```
ai-interview-platform/
├── backend/
│   ├── main.py            FastAPI app, CORS, routers, /health
│   ├── config.py          every setting, read from .env once
│   ├── llm.py             get_llm() and get_embeddings() — the provider switch
│   ├── schemas.py         Pydantic request/response models
│   ├── db.py              SQLite: sessions + qa tables
│   ├── core/              business logic (see above)
│   └── routers/           upload.py, interview.py, report.py
├── frontend/
│   ├── app.py             Streamlit UI — setup, interview, report
│   ├── api_client.py      the only file that knows the backend's URL
│   └── voice.py           gTTS/pyttsx3 TTS, Whisper STT, speaking analysis
├── data/                  uploads, chroma vectors, interview.db  (gitignored)
├── test_pipeline.py       offline tests, no API key required
├── requirements.txt
├── .env.example
└── README.md
```

### API

| Method | Route | Does |
|---|---|---|
| `POST` | `/upload/resume-jd` | parse both docs, extract skills, compute gap, create session, index vectors |
| `GET` | `/interview/{session_id}/next-question` | retrieve context, generate the next adaptive question |
| `POST` | `/interview/answer` | score the answer, update difficulty, store it |
| `GET` | `/report/{session_id}/final` | overall score, per-skill scores, written feedback |
| `DELETE` | `/interview/{session_id}` | delete the session rows and its Chroma directory |
| `GET` | `/health` | which providers are live |

---

## The design decisions, and why

These are the things an interviewer will ask about. Each one has a real reason.

**Why FastAPI over Flask?** Pydantic validation is built in, so a malformed request
fails with a clear 422 at the boundary instead of being hand-checked in every
function. Swagger docs are generated from the schemas, which is how I tested every
endpoint before writing any UI. And it's async-native, which matters because almost
all the request time is spent waiting on the LLM — that's I/O, not CPU.

**Why one `get_llm()` behind everything?** No file outside `llm.py` imports a
provider SDK. Mistral, Gemini, Ollama and a HuggingFace endpoint all satisfy the
same LangChain interface, so switching is one line in `.env`. That gave me free
local iteration while developing and a demo that still runs with no internet.

**Why temperature 0.4?** A trade-off. At 0.0 the questions repeat themselves. High
values make the model drift off-format and break the JSON contract. 0.4 keeps
variety without losing structure.

**Why chunk size 500 with overlap 50?** The overlap means a sentence cut at a chunk
boundary still survives whole inside at least one chunk, so retrieval doesn't return
half an idea.

**Why RAG at all for a two-page resume?** Fair challenge, and I'll concede half of
it: for a resume alone you could just paste the whole thing into the prompt.
Retrieval earns its place by keeping the generator focused on the relevant section
instead of diluting attention over the whole document, and it's the part that still
works when the JD is long or there are several documents.

**Why local embeddings instead of an embeddings API?** Embeddings run on every
single answer we score. An API round-trip there would be the latency bottleneck and
would cost per call. MiniLM is 80MB and fast enough to feel instant.

**Why three scoring signals instead of one?** This was the hardest problem in the
project and the answer I'm most pleased with. There is no answer key — an open-ended
spoken answer can be correct in a hundred phrasings, so there's nothing to diff
against. Instead of hunting for one perfect metric I combined three imperfect ones
that fail differently:

| Signal | Weight | Strength | How it fails |
|---|---|---|---|
| Keyword overlap | 20% | cheap, deterministic, transparent | blind to paraphrase — right idea, different words, score zero |
| Cosine similarity to a model answer | 30% | catches meaning across different wording | confident nonsense using the right vocabulary still scores |
| LLM rubric (correctness, depth, clarity) | 50% | the only signal that can say "that's wrong" | non-deterministic, costs a call |

Weighting them means no single failure mode decides the score. The LLM carries the
most weight because it's the only one that can judge correctness.

**Why cosine and not Euclidean distance?** Cosine measures the angle between two
vectors and ignores magnitude. A long answer and a short answer that say the same
thing point in the same direction but have different lengths — Euclidean would punish
that difference. The question is *what* was said, not how much.

**Why does the question generator also return an ideal answer?** So cosine has
something correct to compare against. An earlier version compared the answer to the
retrieved resume chunks, which measured "does this sound like your resume" rather
than "is this right". Generating a reference answer alongside the question fixed it.

**Why are missing skills asked first?** The gap is where the signal is; matched
skills tell a recruiter less. Counter-argument I'll concede: warming up on strengths
first would probably be a nicer candidate experience.

**Why SQLite and not a dict?** A module-level dict doesn't survive a restart and
breaks the moment uvicorn runs more than one worker, because each worker gets its own
copy. SQLite is a single file with zero configuration and fixes both.

**What if the LLM returns something that isn't valid JSON?** I never assume it won't.
`PydanticOutputParser` puts the exact schema in the prompt and validates the reply,
and every LLM call is wrapped with a fallback — a templated question, a neutral score,
a default summary. A bad response degrades the interview instead of killing it.

---

## Honest limitations

Volunteering these before being asked is worth more than hiding them.

1. **Scoring weights are reasoned, not tuned.** Validating 20/30/50 properly needs a
   set of human-scored answers to fit against, which I didn't have. This is the first
   thing I'd fix with more time.
2. **Scanned resumes return nothing.** `PyPDFLoader` reads the text layer only. An
   image-only PDF needs OCR, which I kept out of scope.
3. **The regex fallback has a small vocabulary.** It only runs when the LLM call
   fails, but when it does, anything outside its ~45 known skills is invisible.
4. **Cleanup is manual.** `DELETE /interview/{session_id}` removes a session's rows
   and its Chroma folder, but nothing calls it automatically. A scheduled sweep of
   sessions older than a day is the real fix.
5. **Whisper adds latency on first use.** The model loads once per process (~140MB
   for `base`), so the first spoken answer is slower than the rest.
6. **The fluency score is reported, not weighted.** Delivery is shown next to the
   content score but doesn't change it. Whether it should is a product decision I
   haven't made.
7. **No authentication.** Anyone with a session ID can read that session's report.

---

## The 90-second explanation

> Interview practice has no feedback loop. Generic question banks ask about things
> that aren't on your resume, and mock interviews with humans don't scale.
>
> So: you upload your resume and a job description. The backend parses both, an LLM
> extracts the skills, and set operations compute the gap — matched, missing, match
> ratio. That gap decides what the interview focuses on, missing skills first.
>
> Both documents get chunked and embedded into a session-scoped Chroma store. For
> each question the system retrieves the chunks most relevant to the target skill
> and passes them to the LLM with the difficulty level and the questions already
> asked, using PydanticOutputParser so the response comes back as a validated
> object — the question, expected keywords, and a model answer.
>
> When you answer, three scorers run: keyword coverage, cosine similarity between
> your answer and that model answer, and an LLM grading correctness, depth and
> clarity. They blend 20/30/50 into one score, and that score moves the next
> question's difficulty up or down.
>
> The decision I'm most pleased with is that no file outside `llm.py` imports a
> provider SDK, so the whole app runs offline on Ollama with no API key by changing
> one line in `.env`.

Then stop and let them follow up.
