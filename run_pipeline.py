import argparse
import os
import sys

def check_env():
    import config
    if not config.GROQ_API_KEY:
        print("❌  GROQ_API_KEY is not set.")
        print("    Export it: export GROQ_API_KEY=gsk_...")
        sys.exit(1)
    print(f"✅  API key set ({config.GROQ_API_KEY[:12]}...)")


def step1():
    print("\n" + "═" * 50)
    print("STEP 1 — Ingest & parse CSV")
    print("═" * 50)
    import config, shutil
    os.makedirs(config.DATA_DIR, exist_ok=True)
    # Try to find the CSV
    for candidate in ["conversations.csv",
                      os.path.join(config.DATA_DIR, "conversations.csv")]:
        if os.path.exists(candidate):
            if candidate != config.CSV_PATH:
                shutil.copy(candidate, config.CSV_PATH)
            break
    else:
        print(f"❌  conversations.csv not found. "
              f"Place it at {config.CSV_PATH}")
        sys.exit(1)

    from importlib import import_module
    mod = import_module("01_ingest")
    mod.ingest(config.CSV_PATH,
               os.path.join(config.DATA_DIR, "messages.json"))


def step2():
    print("\n" + "═" * 50)
    print("STEP 2 — Build RAG (checkpoints + FAISS)")
    print("═" * 50)
    from importlib import import_module
    mod = import_module("02_build_rag")
    mod.main()


def step3():
    print("\n" + "═" * 50)
    print("STEP 3 — Extract user persona")
    print("═" * 50)
    from importlib import import_module
    mod = import_module("03_extract_persona")
    mod.main()


def chat():
    from importlib import import_module
    mod = import_module("04_chatbot")
    mod.cli_loop()


def api():
    from importlib import import_module
    mod = import_module("04_chatbot")
    app = mod.create_flask_app()
    print("[pipeline] Starting Flask API on port 5000 ...")
    app.run(host="0.0.0.0", port=5000, debug=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Persona Pipeline")
    parser.add_argument("--step", type=int, choices=[1, 2, 3],
                        help="Run a specific step only")
    parser.add_argument("--chat", action="store_true",
                        help="Launch chatbot CLI")
    parser.add_argument("--api", action="store_true",
                        help="Launch chatbot Flask API")
    args = parser.parse_args()

    check_env()

    if args.chat:
        chat()
    elif args.api:
        api()
    elif args.step == 1:
        step1()
    elif args.step == 2:
        step2()
    elif args.step == 3:
        step3()
    else:
        # Run all steps
        step1()
        step2()
        step3()
        print("\n✅  Pipeline complete! Run  python run_pipeline.py --chat  to start chatbot.")