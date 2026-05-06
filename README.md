---
title: KaStack

colorFrom: pink
colorTo: red
sdk: docker
pinned: false
---
# Conversation RAG + Persona Chatbot

A scalable RAG system built on FAISS + Sentence Transformers + Claude API.
Processes ~200K messages from a conversations CSV, builds topic checkpoints
via sliding-window LLM comparison, extracts a user persona, and exposes a
chatbot CLI and REST API.

---

## Architecture

```
conversations.csv
      │
      ▼
01_ingest.py          → data/messages.json        (194K flat ordered messages)
      │
      ▼
02_build_rag.py
  ├── Topic Checkpoints (sliding window + LLM)    → checkpoints/topic_checkpoints.json
  ├── 100-msg Checkpoints                         → checkpoints/message_checkpoints.json
  └── FAISS Indexes                               → vector_store/
          ├── summaries.faiss   (topic + chunk summaries)
          └── chunks.faiss      (raw message windows)
      │
      ▼
03_extract_persona.py → persona.json              (habits, traits, style, facts)
      │
      ▼
04_chatbot.py         → CLI or Flask REST API
```

---

## Setup

### 1. Clone / copy files

```
rag_system/
├── conversations.csv       
├── config.py
├── 01_ingest.py
├── 02_build_rag.py
├── 03_extract_persona.py
├── 04_chatbot.py
├── embedder.py
├── retriever.py
├── llm_client.py
├── run_pipeline.py
└── requirements.txt
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note on FAISS**: use `faiss-gpu` instead of `faiss-cpu` if you have a GPU.

### 3. Set your GROQ API key

```bash
export GROQ_API_KEY=sk-ant-...
```

---

## Running the pipeline

### All steps at once
```bash
python run_pipeline.py
```

### Step by step (recommended for large datasets)
```bash
python run_pipeline.py --step 1    # ingest CSV → messages.json
python run_pipeline.py --step 2    # build RAG + FAISS indexes
python run_pipeline.py --step 3    # extract persona
```

Steps 2 and 3 are **resumable** — if interrupted, they skip already-completed
work on re-run. Safe to re-run at any time.

---

## Using the chatbot

### CLI
```bash
python run_pipeline.py --chat
# or
python 04_chatbot.py
```

### REST API
```bash
python run_pipeline.py --api
# or
python 04_chatbot.py --api --port 5000
```

**Endpoints:**
```
POST /ask          body: {"query": "What are this user's habits?"}
POST /reset        clears conversation history
GET  /persona      returns full persona JSON
GET  /health       health check
```

**Example curl:**
```bash
curl -X POST http://localhost:5000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "What kind of person is this user?"}'
```

---

## Configuration (config.py)

| Parameter | Default | Description |
|---|---|---|
| `SLIDING_WINDOW_SIZE` | 15 | Messages per window for topic detection |
| `TOPIC_OVERLAP` | 3 | Overlap between adjacent windows |
| `MSG_CHUNK_SIZE` | 100 | Messages per 100-msg checkpoint |
| `TOP_K_SUMMARIES` | 4 | Summaries returned per query |
| `TOP_K_CHUNKS` | 6 | Raw chunks returned per query |
| `EMBED_MODEL` | all-MiniLM-L6-v2 | Sentence-transformer model |
| `SUMMARY_MODEL` | claude-haiku-4-5 | Model for bulk summarisation |
| `CHAT_MODEL` | claude-sonnet-4-20250514 | Model for chatbot responses |

---

## Example questions to ask

- *"What kind of person is this user?"*
- *"What are their habits?"*
- *"How do they communicate? What's their tone like?"*
- *"What do we know about their family or relationships?"*
- *"What are their hobbies and interests?"*
- *"What major life events have they mentioned?"*
- *"Are they introverted or extroverted?"*

---

## Performance notes

- **Step 1** (ingest): ~30 seconds for 11K rows
- **Step 2** (build RAG): the topic detection pass makes ~13K LLM calls
  (haiku, cheap). Budget ~2–3 hours + ~$2–4 at haiku pricing.
  Run overnight. Results are cached — safe to interrupt and resume.
- **Step 3** (persona): ~20 LLM calls. < 1 minute.
- **Chatbot**: each query = 1 embedding call (local) + 1 FAISS search + 1 LLM call.

---

## File outputs

| File | Contents |
|---|---|
| `data/messages.json` | All 194K messages, flat, ordered, with metadata |
| `checkpoints/topic_checkpoints.json` | Topic segments + summaries |
| `checkpoints/message_checkpoints.json` | 100-msg chunks + summaries |
| `vector_store/summaries.faiss` | FAISS index of all summaries |
| `vector_store/summaries_meta.pkl` | Metadata parallel to FAISS rows |
| `vector_store/chunks.faiss` | FAISS index of raw message windows |
| `vector_store/chunks_meta.pkl` | Metadata for raw chunks |
| `persona.json` | Extracted user persona (habits, traits, etc.) |


Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
