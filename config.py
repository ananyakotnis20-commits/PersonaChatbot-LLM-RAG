"""""
config.py - Central
configuration
for the RAG system.
Edit
these
values
to
tune
behavior.
"""
import os
from dotenv import load_dotenv
load_dotenv()
# ── Paths ──────────────────────────────────────────────────────────────────
DATA_DIR         = "data"
CSV_PATH         = os.path.join(DATA_DIR, "conversations.csv")
CHECKPOINTS_DIR  = "checkpoints"
VECTOR_STORE_DIR = "vector_store"
PERSONA_PATH     = "persona.json"

# ── Model settings ─────────────────────────────────────────────────────────
# Embedding model (local, no API key needed)
EMBED_MODEL      = "all-MiniLM-L6-v2"   # fast + good quality; ~80MB download
EMBED_DIM        = 384                  # dimension for all-MiniLM-L6-v2

SUMMARY_MODEL = "llama-3.1-8b-instant"   # fast, free
CHAT_MODEL    = "llama-3.3-70b-versatile" # powerful, free




# ── RAG parameters ─────────────────────────────────────────────────────────
SLIDING_WINDOW_SIZE     = 15    # messages per window for topic detection
TOPIC_OVERLAP           = 3     # overlap between windows
MSG_CHUNK_SIZE          = 100   # messages per 100-message checkpoint
TOPIC_CHANGE_THRESHOLD  = 0.35  # cosine distance above this → topic changed
TOP_K_SUMMARIES         = 4     # summaries retrieved per query
TOP_K_CHUNKS            = 6     # raw message chunks retrieved per query
MAX_CONTEXT_TOKENS      = 3000  # cap context fed to chat LLM

# ── Persona extraction ─────────────────────────────────────────────────────
PERSONA_BATCH_SIZE      = 50    # topic summaries per persona extraction batch

# ── API ────────────────────────────────────────────────────────────────────
GROQ_API_KEY  = os.environ.get("GROQ_API_KEY", "")