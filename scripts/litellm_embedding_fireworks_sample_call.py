import os

import litellm

FIREWORKS_MODEL = "nomic-ai/nomic-embed-text-v1.5"
FIREWORKS_API_BASE = "https://api.fireworks.ai/inference/v1"


def _get_fireworks_api_key() -> str:
    api_key = os.getenv("FIREWORKS_API_KEY") or os.getenv("FIREWORKS_AI_API_KEY")
    if not api_key:
        raise RuntimeError("Set FIREWORKS_API_KEY or FIREWORKS_AI_API_KEY before running this script.")
    return api_key


def main() -> None:
    input_text = "Sample text for embedding"
    response = litellm.embedding(
        model=FIREWORKS_MODEL,
        input=input_text,
        api_base=FIREWORKS_API_BASE,
        api_key=_get_fireworks_api_key(),
        custom_llm_provider="openai",
    )
    embedding = response["data"][0]["embedding"]
    print(f"Received embedding of length {len(embedding)}")


if __name__ == "__main__":
    main()
