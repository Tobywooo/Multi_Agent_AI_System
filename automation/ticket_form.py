"""Playwright workflow for the mock support ticket form.

Opens the app, fills Issue Description / Category / Resolution Notes, submits,
and verifies the confirmation panel shows what we entered.

Elements are located by their accessible label or role (get_by_label,
get_by_role) rather than CSS ids, so the script interacts with the page the
way a user or screen reader would.

Usage (standalone test, no agents involved):
    python -m automation.ticket_form --headed
    python -m automation.ticket_form --category Email   # unsupported-category failure
"""

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import Error as PlaywrightError
from playwright.sync_api import Page, expect, sync_playwright

from automation.app_server import serve_mock_app

TICKET_ID_PATTERN = re.compile(r"^\d{5}$")
UI_TIMEOUT_MS = 5_000


class TicketSubmissionError(Exception):
    """The ticket could not be submitted or the confirmation did not verify."""


class UnsupportedCategoryError(TicketSubmissionError):
    """The requested category is not one of the form's dropdown options."""


@dataclass
class TicketResult:
    ticket_id: str
    category: str
    resolution: str
    url: str


def _normalize(text: str) -> str:
    # The confirmation renders text inline, so newlines collapse to spaces.
    return " ".join(text.split())


def _resolve_category_label(page: Page, category: str) -> str:
    """Match a category name (e.g. 'hardware', 'Account Access') to a dropdown option label."""
    options = page.get_by_label("Category").locator("option")
    labels = [label.strip() for label in options.all_inner_texts()]
    values = [options.nth(i).get_attribute("value") or "" for i in range(options.count())]

    wanted = category.strip().lower()
    for label, value in zip(labels, values):
        if value and wanted in (label.lower(), value.lower()):
            return label

    available = ", ".join(label for label, value in zip(labels, values) if value)
    raise UnsupportedCategoryError(
        f"Category '{category}' is not available in the form (options: {available})"
    )


def fill_and_submit(page: Page, url: str, issue: str, category: str, resolution: str) -> TicketResult:
    page.goto(url)
    expect(page.get_by_role("heading", name="Submit a Support Ticket")).to_be_visible(
        timeout=UI_TIMEOUT_MS
    )

    category_label = _resolve_category_label(page, category)
    page.get_by_label("Issue Description").fill(issue)
    page.get_by_label("Category").select_option(label=category_label)
    page.get_by_label("Resolution Notes").fill(resolution)
    page.get_by_role("button", name="Submit Ticket").click()

    error = page.get_by_role("alert")
    if error.is_visible():
        raise TicketSubmissionError(f"Form rejected the ticket: {error.inner_text().strip()}")

    confirmation = page.get_by_role("status")
    try:
        expect(confirmation).to_be_visible(timeout=UI_TIMEOUT_MS)
        expect(confirmation).to_contain_text("Ticket Submitted Successfully!")
    except AssertionError as exc:
        raise TicketSubmissionError("Confirmation did not appear after submitting") from exc

    ticket_id = page.locator("#ticket-id").inner_text().strip()
    shown_category = page.locator("#ticket-category").inner_text().strip()
    shown_resolution = page.locator("#ticket-resolution").inner_text().strip()

    problems = []
    if not TICKET_ID_PATTERN.match(ticket_id):
        problems.append(f"ticket ID '{ticket_id}' is not a 5-digit number")
    if shown_category != category_label:
        problems.append(f"category shows '{shown_category}', expected '{category_label}'")
    if _normalize(shown_resolution) != _normalize(resolution):
        problems.append("resolution shown does not match what was entered")
    if problems:
        raise TicketSubmissionError("Verification failed: " + "; ".join(problems))

    return TicketResult(ticket_id, shown_category, shown_resolution, url)


def submit_ticket(
    issue: str,
    category: str,
    resolution: str,
    *,
    headless: bool = True,
    slow_mo: int = 0,
    screenshot_path: Path | None = None,
    hold_seconds: float = 0,
) -> TicketResult:
    """Serve the mock app, submit one ticket, verify it, and return the confirmation details.

    hold_seconds keeps the confirmation on screen before the browser closes (for demos).
    Raises TicketSubmissionError (or UnsupportedCategoryError) on any failure.
    """
    with serve_mock_app() as url, sync_playwright() as p:
        try:
            browser = p.chromium.launch(headless=headless, slow_mo=slow_mo)
        except PlaywrightError as exc:
            # Usually a missing browser; Playwright's own message is a long banner.
            first_line = exc.message.strip().splitlines()[0]
            raise TicketSubmissionError(
                f"Could not launch the browser ({first_line}). "
                "Run `playwright install chromium` and try again."
            ) from exc
        page = browser.new_page()
        try:
            result = fill_and_submit(page, url, issue, category, resolution)
            if hold_seconds:
                page.wait_for_timeout(hold_seconds * 1000)
            return result
        except PlaywrightError as exc:
            raise TicketSubmissionError(f"Browser automation error: {exc.message}") from exc
        finally:
            if screenshot_path:
                screenshot_path.parent.mkdir(parents=True, exist_ok=True)
                page.screenshot(path=str(screenshot_path), full_page=True)
            browser.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Submit one ticket to the mock support app.")
    parser.add_argument("--issue", default="My keyboard has stopped working.")
    parser.add_argument("--category", default="Hardware")
    parser.add_argument(
        "--resolution",
        default="Reconnect the device and verify that the computer recognizes it.",
    )
    parser.add_argument("--headed", action="store_true", help="show the browser window")
    parser.add_argument("--slow-mo", type=int, default=0, help="ms delay between actions")
    parser.add_argument("--screenshot", type=Path, help="save a screenshot of the final page")
    args = parser.parse_args()

    try:
        result = submit_ticket(
            args.issue,
            args.category,
            args.resolution,
            headless=not args.headed,
            slow_mo=args.slow_mo,
            screenshot_path=args.screenshot,
        )
    except TicketSubmissionError as exc:
        print(f"FAILED: {exc}")
        raise SystemExit(1)
    print(f"Ticket #{result.ticket_id} submitted and verified ({result.category})")


if __name__ == "__main__":
    main()
