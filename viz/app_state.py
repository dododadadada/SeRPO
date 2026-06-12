"""Derive a mock-app UI state from a single trajectory step.

Pure functions: derive_ui_state(app, api, output, prev) -> dict.
Mapped for venmo + phone; everything else falls back to a generic 'api_call' state.
A step that calls no API (app is None) leaves the screen unchanged (returns prev).
"""
import json
from typing import Any


def _try_json(output: str) -> Any:
    try:
        return json.loads(output)
    except (ValueError, TypeError):
        return None


def derive_ui_state(app: str | None, api: str | None, output: str,
                    prev: dict | None) -> dict:
    # Pure-compute / no-API step: screen does not change.
    if app is None:
        return prev if prev is not None else {"kind": "idle", "title": ""}

    if app == "api_docs":
        return {"kind": "docs", "title": "📖 Reading API docs"}

    if api == "complete_task":
        return {"kind": "result", "title": "✅ Task submitted"}

    if app == "venmo":
        return _venmo(api, output, prev)
    if app == "phone":
        return _phone(api, output, prev)

    # Generic fallback for any unmapped (app, api).
    return {"kind": "api_call", "title": f"⚙️ {app}.{api}()"}


def _venmo(api: str | None, output: str, prev: dict | None) -> dict:
    if api == "login":
        return {"kind": "login", "title": "Venmo — signed in"}
    if api == "show_transactions":
        data = _try_json(output)
        rows = []
        if isinstance(data, list):
            for tx in data:
                rows.append({
                    "amount": tx.get("amount"),
                    "description": tx.get("description", ""),
                    "created_at": tx.get("created_at", ""),
                    "sender": (tx.get("sender") or {}).get("name", ""),
                    "receiver": (tx.get("receiver") or {}).get("name", ""),
                })
        return {"kind": "venmo_transactions", "title": "Venmo — transactions",
                "rows": rows}
    return {"kind": "api_call", "title": f"⚙️ venmo.{api}()"}


def _phone(api: str | None, output: str, prev: dict | None) -> dict:
    # Implemented in Task 3.
    return {"kind": "api_call", "title": f"⚙️ phone.{api}()"}
