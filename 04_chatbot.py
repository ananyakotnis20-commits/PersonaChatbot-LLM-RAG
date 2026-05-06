from __future__ import annotations
import argparse
import json
import os
import sys

import config
import llm_client
from retriever import Retriever



SYSTEM_TEMPLATE = """You are a helpful assistant that has deeply analysed
a large set of conversations involving User 1. You have access to:

1. A structured persona profile of User 1
2. Relevant conversation excerpts and summaries retrieved for this query

Answer questions about User 1 based ONLY on the evidence provided.
Be specific and quote or reference examples when possible.
If the evidence does not support an answer, say so clearly.

## User 1 Persona Profile
{persona}

## Retrieved Context
{context}
"""


#CHATBOT CLASS
class PersonaChatbot:
    def __init__(self):
        self.retriever = Retriever().load()
        self.persona = self._load_persona()
        self.history: list[dict] = []  # multi-turn conversation history

    def _load_persona(self) -> dict:
        if os.path.exists(config.PERSONA_PATH):
            with open(config.PERSONA_PATH) as f:
                return json.load(f)
        print("[chatbot] WARNING: persona.json not found. "
              "Run 03_extract_persona.py first.")
        return {}

    def _build_system(self, context_text: str) -> str:
        persona_str = json.dumps(self.persona, indent=2)[:2000]
        return SYSTEM_TEMPLATE.format(
            persona=persona_str,
            context=context_text,
        )

    def ask(self, user_query: str) -> str:
        """Single-turn ask (with history for multi-turn)."""
        # Retrieve relevant context
        result = self.retriever.retrieve(user_query)
        context_text = result["context_text"]

        # Build messages array (multi-turn aware)
        system_prompt = self._build_system(context_text)
        self.history.append({"role": "user", "content": user_query})

        response = llm_client.chat(
            messages=self.history,
            system=system_prompt,
            max_tokens=1024,
        )

        self.history.append({"role": "assistant", "content": response})

        # Keep history from growing unbounded (last 10 turns)
        if len(self.history) > 20:
            self.history = self.history[-20:]

        return response

    def reset(self):
        self.history = []



# CLI mode

WELCOME = """
╔══════════════════════════════════════════════════════════╗
║         Conversation Persona RAG Chatbot  🤖             ║
╠══════════════════════════════════════════════════════════╣
║  Ask anything about User 1's habits, personality, life  ║
║  Type  /reset  to clear history                         ║
║  Type  /quit   to exit                                  ║
╚══════════════════════════════════════════════════════════╝

Example questions:
  • What kind of person is this user?
  • What are their habits?
  • How do they talk / what is their communication style?
  • What do we know about their relationships?
  • What are their hobbies or interests?
"""


def cli_loop():
    print(WELCOME)
    bot = PersonaChatbot()
    while True:
        try:
            query = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not query:
            continue
        if query.lower() == "/quit":
            print("Goodbye!")
            break
        if query.lower() == "/reset":
            bot.reset()
            print("[History cleared]")
            continue

        print("\nAssistant: ", end="", flush=True)
        try:
            answer = bot.ask(query)
            print(answer)
        except Exception as e:
            print(f"\n[Error: {e}]")



# Flask API mode


def create_flask_app() -> "Flask":
    from flask import Flask, request, jsonify
    app = Flask(__name__)
    bot = PersonaChatbot()

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "ok"})

    @app.route("/ask", methods=["POST"])
    def ask():
        data = request.get_json(force=True)
        query = data.get("query", "").strip()
        if not query:
            return jsonify({"error": "query is required"}), 400
        try:
            answer = bot.ask(query)
            return jsonify({
                "query": query,
                "answer": answer,
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/reset", methods=["POST"])
    def reset():
        bot.reset()
        return jsonify({"status": "history cleared"})

    @app.route("/persona", methods=["GET"])
    def persona():
        return jsonify(bot.persona)

    return app



# Entry point


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--api", action="store_true",
                        help="Run as Flask REST API instead of CLI")
    parser.add_argument("--port", type=int, default=5000)
    args = parser.parse_args()

    if args.api:
        app = create_flask_app()
        print(f"[chatbot] Starting Flask API on port {args.port} ...")
        app.run(host="0.0.0.0", port=args.port, debug=False)
    else:
        cli_loop()
