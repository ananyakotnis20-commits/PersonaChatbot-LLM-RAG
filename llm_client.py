"""
llm_client.py - Thin wrapper around the OpenAI API.
"""

import time
import groq
import config

_client = None

def get_client():
    global _client
    if _client is None:
        _client = groq.Groq(api_key=config.GROQ_API_KEY)
    return _client

def complete(
    prompt: str,
    system: str = "",
    model: str | None = None,
    max_tokens: int = 1024,
    retries: int = 6,
) -> str:
    model = model or config.SUMMARY_MODEL
    client = get_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for attempt in range(retries):
        try:
            resp = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                messages=messages,
            )
            return resp.choices[0].message.content.strip()
        except groq.RateLimitError:
            wait = 5* (2**attempt)
            print(f"  [llm] Rate limit – retrying in {wait}s ...")
            time.sleep(wait)
        except groq.APIStatusError:
            if attempt == retries - 1:
                raise
            time.sleep(2)
    raise RuntimeError("LLM call failed after retries")


def chat(
    messages: list[dict],
    system: str = "",
    model: str | None = None,
    max_tokens: int = 1024,
) -> str:
    model = model or config.CHAT_MODEL
    client = get_client()
    all_messages = []
    if system:
        all_messages.append({"role": "system", "content": system})
    all_messages.extend(messages)

    resp = client.chat.completions.create(
        model=model,
        max_tokens=max_tokens,
        messages=all_messages,
    )
    return resp.choices[0].message.content.strip()