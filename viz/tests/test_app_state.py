import json
from viz.app_state import derive_ui_state


def test_fallback_for_unmapped_api():
    st = derive_ui_state(app="venmo", api="like_transaction", output="{}", prev=None)
    assert st["kind"] == "api_call"
    assert "like_transaction" in st["title"]


def test_api_docs_is_docs_state():
    st = derive_ui_state(app="api_docs", api="show_api_descriptions", output="[]", prev=None)
    assert st["kind"] == "docs"


def test_none_app_keeps_prev_state():
    prev = {"kind": "login", "title": "Venmo"}
    st = derive_ui_state(app=None, api=None, output="2023-01-01", prev=prev)
    assert st == prev  # a pure-compute step doesn't change the app screen


def test_venmo_login():
    out = '{"access_token": "TOK", "token_type": "Bearer"}'
    st = derive_ui_state(app="venmo", api="login", output=out, prev=None)
    assert st["kind"] == "login"
    assert st["title"].lower().startswith("venmo")


def test_venmo_show_transactions_renders_rows():
    out = json.dumps([
        {"amount": 50.0, "description": "Electricity bill",
         "created_at": "2023-03-01T00:00:00",
         "sender": {"name": "Jose"}, "receiver": {"name": "PowerCo"}},
        {"amount": 12.0, "description": "Coffee",
         "created_at": "2023-03-02T00:00:00",
         "sender": {"name": "Jose"}, "receiver": {"name": "Cafe"}},
    ])
    st = derive_ui_state(app="venmo", api="show_transactions", output=out, prev=None)
    assert st["kind"] == "venmo_transactions"
    assert len(st["rows"]) == 2
    assert st["rows"][0]["amount"] == 50.0
    assert st["rows"][0]["description"] == "Electricity bill"


def test_complete_task_is_result():
    st = derive_ui_state(app="supervisor", api="complete_task",
                         output="Execution successful.", prev=None)
    assert st["kind"] == "result"


def test_phone_show_alarms_renders_rows():
    out = json.dumps([
        {"alarm_id": 1, "label": "Weekend wake up", "time": "08:00",
         "snooze_minutes": 5, "enabled": True},
        {"alarm_id": 2, "label": "Workday", "time": "06:30",
         "snooze_minutes": 10, "enabled": True},
    ])
    st = derive_ui_state(app="phone", api="show_alarms", output=out, prev=None)
    assert st["kind"] == "phone_alarms"
    assert len(st["rows"]) == 2
    assert st["rows"][0]["label"] == "Weekend wake up"
    assert st["rows"][0]["snooze_minutes"] == 5


def test_phone_update_alarm_marks_changed():
    prev = {"kind": "phone_alarms", "title": "Phone — alarms", "rows": [
        {"alarm_id": 1, "label": "Weekend wake up", "time": "08:00",
         "snooze_minutes": 5, "enabled": True},
    ]}
    out = json.dumps({"alarm_id": 1, "label": "Weekend wake up",
                      "time": "08:00", "snooze_minutes": 15, "enabled": True})
    st = derive_ui_state(app="phone", api="update_alarm", output=out, prev=prev)
    assert st["kind"] == "phone_alarms"
    row = next(r for r in st["rows"] if r["alarm_id"] == 1)
    assert row["snooze_minutes"] == 15
    assert row["changed"] is True


def test_venmo_show_transactions_bad_json_yields_empty_rows():
    st = derive_ui_state(app="venmo", api="show_transactions", output="not json", prev=None)
    assert st["kind"] == "venmo_transactions"
    assert st["rows"] == []
