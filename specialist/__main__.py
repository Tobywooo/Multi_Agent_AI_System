"""Run the Specialist Agent server.

Usage:
    python -m specialist               # http://127.0.0.1:8001
    python -m specialist --port 9001
"""

import argparse
import logging

import uvicorn

from a2a_protocol.server import create_app
from specialist.agent import AGENT_CARD, handle_task, warm_up

DEFAULT_PORT = 8001


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Specialist Agent.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    # Only our task-status log at INFO; library chatter (httpx, HF Hub) stays quiet.
    logging.basicConfig(level=logging.WARNING, format="[Specialist] %(message)s")
    logging.getLogger("a2a.server").setLevel(logging.INFO)
    print("[Specialist] Loading knowledge base...")
    warm_up()

    app = create_app({**AGENT_CARD, "url": f"http://127.0.0.1:{args.port}"}, handle_task)
    uvicorn.run(app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
