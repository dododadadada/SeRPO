"""Derive a mock-app UI state from a single trajectory step.

Pure functions: derive_ui_state(app, api, output, prev, code="") -> dict.
Mapped for venmo + phone; everything else falls back to a generic 'api_call' state.
A step that calls no API (app is None) leaves the screen unchanged (returns prev).
"""
import json
import re
from typing import Any

_IDLE: dict = {"kind": "idle", "title": ""}

_ASSIGN_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*=\s*(-?\d+)\s*$", re.MULTILINE)
_KWARG_RE_TMPL = r"(?<!\w){name}\s*=\s*([A-Za-z_]\w*|-?\d+)"


def _resolve_int_kwarg(code: str, kwarg: str) -> int | None:
    """Find `kwarg=<value>` in code; value may be an int literal or a variable
    name previously assigned `var = <int>`. Returns the int or None."""
    if not code:
        return None
    assigns = {m.group(1): int(m.group(2)) for m in _ASSIGN_RE.finditer(code)}
    m = re.search(_KWARG_RE_TMPL.format(name=re.escape(kwarg)), code)
    if not m:
        return None
    token = m.group(1)
    if re.fullmatch(r"-?\d+", token):
        return int(token)
    return assigns.get(token)


def _parse_answer(code: str):
    """Extract the answer=... literal from a complete_task(...) call. Returns a
    string form suitable for display, or None. Handles numbers and quoted strings."""
    if not code:
        return None
    # answer is always single-line; no re.DOTALL so a lazy match can't span
    # newlines into a later complete_task call. Stop at the first comma/paren/EOL.
    m = re.search(r"answer\s*=\s*([^,)\n]+)", code)
    if not m:
        return None
    val = m.group(1).strip()
    # strip surrounding quotes if a string literal
    if (val.startswith("'") and val.endswith("'")) or (val.startswith('"') and val.endswith('"')):
        val = val[1:-1]
    return val


def _try_json(output: str) -> Any:
    try:
        return json.loads(output)
    except (ValueError, TypeError):
        return None


def derive_ui_state(app: str | None, api: str | None, output: str,
                    prev: dict | None, code: str = "") -> dict:
    # Pure-compute / no-API step: screen does not change.
    if app is None:
        return prev if prev is not None else dict(_IDLE)
    if api is None:
        return prev if prev is not None else dict(_IDLE)

    if app == "api_docs":
        return {"kind": "docs", "title": "📖 Reading API docs"}

    if api == "complete_task":
        ans = _parse_answer(code)
        st = {"kind": "result", "title": "✅ Task submitted"}
        if ans is not None and ans != "None":
            st["answer"] = ans
        return st

    if app == "venmo":
        return _venmo(api, output, prev)
    if app == "phone":
        return _phone(api, output, prev, code)

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


def _phone(api: str | None, output: str, prev: dict | None, code: str = "") -> dict:
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
        prev_rows = (prev or {}).get("rows", []) if isinstance(prev, dict) else []
        rows = [dict(r) for r in prev_rows]  # shallow copy

        # Parse alarm_id and snooze_minutes from code (not env output).
        target_id = _resolve_int_kwarg(code, "alarm_id")
        new_snooze = _resolve_int_kwarg(code, "snooze_minutes")

        # Reset changed flag on all rows, then update the matched row.
        for r in rows:
            r["changed"] = False
            if target_id is not None and r.get("alarm_id") == target_id:
                if new_snooze is not None:
                    r["snooze_minutes"] = new_snooze
                r["changed"] = True

        return {"kind": "phone_alarms", "title": "Phone — alarms", "rows": rows}
    return {"kind": "api_call", "title": f"⚙️ phone.{api or '?'}()"}
