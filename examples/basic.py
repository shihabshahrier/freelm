"""Run me:  python examples/basic.py

Set at least one of OPENROUTER_API_KEY / GEMINI_API_KEY / NVIDIA_API_KEY first.
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
