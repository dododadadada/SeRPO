"""Derive a mock-app UI state from a single trajectory step.

Pure functions: derive_ui_state(app, api, output, prev) -> dict.
Mapped for venmo + phone; everything else falls back to a generic 'api_call' state.
A step that calls no API (app is None) leaves the screen unchanged (returns prev).
"""
import json
from typing import Any

_IDLE: dict = {"kind": "idle", "title": ""}


def _try_json(output: str) -> Any:
    try:
        return json.loads(output)
    except (ValueError, TypeError):
        return None


def derive_ui_state(app: str | None, api: str | None, output: str,
                    prev: dict | None) -> dict:
    # Pure-compute / no-API step: screen does not change.
    if app is None:
        return prev if prev is not None else dict(_IDLE)
    if api is None:
        return prev if prev is not None else dict(_IDLE)

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
    if api == "login":
        return {"kind": "login", "title": "Phone — signed in"}
    if api == "show_alarms":
        data = _try_json(output)
        rows = []
        if isinstance(data, list):
            for a in data:
                rows.append({
                    "alarm_id": a.get("alarm_id"),
                    "label": a.get("label", ""),
                    "time": a.get("time", ""),
                    "snooze_minutes": a.get("snooze_minutes"),
                    "enabled": a.get("enabled", True),
                    "changed": False,
                })
        return {"kind": "phone_alarms", "title": "Phone — alarms", "rows": rows}
    if api == "update_alarm":
        updated = _try_json(output)
        prev_rows = (prev or {}).get("rows", []) if isinstance(prev, dict) else []
        rows = [dict(r) for r in prev_rows]  # shallow copy
        if isinstance(updated, dict):
            uid = updated.get("alarm_id")
            found = False
            for r in rows:
                r["changed"] = False
                if r.get("alarm_id") == uid:
                    r["snooze_minutes"] = updated.get("snooze_minutes",
                                                       r.get("snooze_minutes"))
                    r["changed"] = True
                    found = True
            if not found and uid is not None:
                rows.append({
                    "alarm_id": uid, "label": updated.get("label", ""),
                    "time": updated.get("time", ""),
                    "snooze_minutes": updated.get("snooze_minutes"),
                    "enabled": updated.get("enabled", True), "changed": True,
                })
        return {"kind": "phone_alarms", "title": "Phone — alarms", "rows": rows}
    return {"kind": "api_call", "title": f"⚙️ phone.{api or '?'}()"}
