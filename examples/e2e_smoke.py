"""Live end-to-end smoke test — reads keys ONLY from the environment.

Never hardcode keys. Load your .env first, then run:

    set -a; . ./.env; set +a
    python examples/e2e_smoke.py

Exercises discovery, chat, failover, streaming (sync + async), and health
against whichever providers you have keys for.
"""
import asyncio

import freelm
from freelm import AsyncFreeLLM, FreeLLM, list_free_models, providers_from_env


def main() -> None:
    print("freelm", freelm.__version__)
    provs = providers_from_env()
    print("providers from env:", [p.name for p in provs])

    try:
        print("discovered free models (OpenRouter):", len(list_free_models(refresh=True)))
    except Exception as e:  # noqa: BLE001 - smoke test, report and continue
        print("discovery skipped:", type(e).__name__, e)

    with FreeLLM(provs, strategy="quota_aware", timeout=20) as llm:
        # thinking models (e.g. gemini-2.5-flash) can spend a tiny budget
        # entirely on reasoning -> empty text, so give them headroom
        r = llm.chat("Reply with exactly one word: pong", max_tokens=128, temperature=0)
        print(f"chat   -> {r.provider}/{r.model}: {r.text!r}")

        print("stream ->", end=" ", flush=True)
        for chunk in llm.stream("Count to five.", max_tokens=128, temperature=0):
            print(chunk, end="", flush=True)
        print()

        print("health:")
        for row in llm.health():
            print(f"   {row['provider']:11} ready={row['ready']!s:5} last_error={row['last_error']}")

    async def _astream() -> None:
        async with AsyncFreeLLM(providers_from_env(), timeout=20) as llm:
            out = ""
            async for chunk in llm.astream("Say hello.", max_tokens=128, temperature=0):
                out += chunk
            print("astream ->", repr(out))

    asyncio.run(_astream())
    print("done.")


if __name__ == "__main__":
    main()
