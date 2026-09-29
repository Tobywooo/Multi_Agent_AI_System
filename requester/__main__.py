"""Run the Requester Agent. Start the Specialist first (`python -m specialist`).

Usage:
    python -m requester                                   # interactive session
    python -m requester "My keyboard has stopped working."  # single request
    python -m requester "..." --headless                  # hidden, fast browser
    python -m requester "..." --simulate slow --timeout 10  # timeout demo
    python -m requester "..." --simulate fail               # failed-task demo

In an interactive session, start a line with /slow or /fail to simulate a
timeout or failure for that request, e.g. `/slow My laptop won't connect to Wi-Fi.`
"""

import argparse
import sys

from requester.agent import DEFAULT_SPECIALIST_URL, log, run

SIMULATE_PREFIXES = {"/slow": "slow", "/fail": "fail"}
QUIT_WORDS = {"", "quit", "exit", "q"}


def parse_line(line: str) -> tuple[str, str | None]:
    """Split an interactive line into (request, simulate mode)."""
    command, _, rest = line.partition(" ")
    if command.lower() in SIMULATE_PREFIXES:
        return rest.strip(), SIMULATE_PREFIXES[command.lower()]
    return line, None


def interactive(args: argparse.Namespace) -> None:
    print("Requester Agent ready. Specialist:", args.specialist_url)
    print("Type a support request and press Enter. Blank line or 'quit' to exit.\n")

    filed, total = 0, 0
    while True:
        try:
            line = input("Request> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if line.lower() in QUIT_WORDS:
            break

        request, simulate = parse_line(line)
        if not request:
            print("Enter a request after the command, e.g. /slow My laptop won't connect to Wi-Fi.\n")
            continue

        total += 1
        try:
            ticket = run(request, simulate=simulate, **run_options(args))
        except KeyboardInterrupt:
            log("Cancelled.")
            ticket = None
        except Exception as exc:
            # Expected failures are already reported by run(); this catches anything
            # unforeseen so one bad request never ends the session.
            log(f"UNEXPECTED ERROR: {type(exc).__name__}: {exc}. No ticket was submitted.")
            ticket = None
        filed += ticket is not None
        print()

    log(f"Session ended: {filed} of {total} request(s) filed as tickets.")


def run_options(args: argparse.Namespace) -> dict:
    return {
        "specialist_url": args.specialist_url,
        "timeout": args.timeout,
        "poll_interval": args.poll_interval,
        "headed": not args.headless,
        "slow_mo": 0 if args.headless else args.slow_mo,
        "hold_seconds": 0 if args.headless else args.hold,
    }


def main() -> None:
    # Never crash on a character the console can't encode (Windows cp1252).
    sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(description="Run the Requester Agent.")
    parser.add_argument("request", nargs="?", help="a support request; omit for an interactive session")
    parser.add_argument("--specialist-url", default=DEFAULT_SPECIALIST_URL)
    parser.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for the Specialist")
    parser.add_argument("--poll-interval", type=float, default=1.0)
    parser.add_argument("--simulate", choices=["slow", "fail"], help="ask the Specialist to misbehave")
    parser.add_argument("--headless", action="store_true", help="hide the browser and skip demo pauses")
    parser.add_argument("--slow-mo", type=int, default=300, help="ms delay between browser actions")
    parser.add_argument("--hold", type=float, default=3.0, help="seconds to keep the confirmation visible")
    parser.add_argument("--headed", action="store_true", help=argparse.SUPPRESS)  # now the default
    args = parser.parse_args()

    if args.request is None:
        interactive(args)
        return

    ticket = run(args.request, simulate=args.simulate, **run_options(args))
    raise SystemExit(0 if ticket else 1)


if __name__ == "__main__":
    main()
