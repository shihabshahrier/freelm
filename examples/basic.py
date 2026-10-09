"""Run me:  python examples/basic.py

Set at least one free key first (GEMINI_API_KEY, GROQ_API_KEY, OPENROUTER_API_KEY, ...);
`freelm doctor` shows which ones work. No keys yet? Run with FREELM_KEYLESS=1 to try
the keyless public endpoints (low limits).
"""
from freelm import FreeLLM


def main() -> None:
    # Reads keys + tiers from environment.
    llm = FreeLLM.from_env(strategy="quota_aware")

    print(llm.text("Explain prompt caching in one sentence.", model="chat:fast"))

    print("\n--- key health ---")
    for row in llm.health():
        print(row)

    llm.close()


if __name__ == "__main__":
    main()
