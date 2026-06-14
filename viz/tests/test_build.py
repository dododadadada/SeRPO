from viz.build_trajectory_json import (
    passed_from_report, build_step_records, _error_summary)


def test_error_summary_detects_401():
    out = ('Execution failed. Traceback:\n  ...\nException: Response status code '
           'is 401: {"message":"You are not authorized or your token is missing."}')
    s = _error_summary(out)
    assert s is not None
    assert s.startswith("401:")


def test_error_summary_none_for_clean_output():
    assert _error_summary('{"access_token": "TOK"}') is None


def test_build_step_records_attaches_root_cause_to_matching_step():
    io = ("\n### Environment Interaction 1\n---\n```python\nx = apis.venmo.login()\n```\n\n```\nok\n```\n"
          "\n### Environment Interaction 2\n---\n```python\ny = apis.venmo.show_transactions()\n```\n\n```\n7134.0\n```\n")
    recs = build_step_records(io, {"step": 2, "text": "missing filter"})
    assert recs[0]["root_cause"] is None
    assert recs[1]["root_cause"] == "missing filter"


def test_build_step_records_flags_error_step():
    io = ("\n### Environment Interaction 1\n---\n```python\n"
          "a = apis.phone.show_alarms()\n```\n\n```\n"
          "Execution failed. Traceback: Exception: status code is 401: "
          '{"message":"missing token"}\n```\n')
    rec = build_step_records(io)[0]
    assert rec["is_error"] is True
    assert rec["error"] is not None


def test_passed_from_report_true():
    md = "Num Passed Tests : 5\nNum Failed Tests : 0\nNum Total  Tests : 5\n"
    assert passed_from_report(md) is True


def test_passed_from_report_false():
    md = "Num Passed Tests : 4\nNum Failed Tests : 1\nNum Total  Tests : 5\n"
    assert passed_from_report(md) is False


def test_passed_from_report_none_when_line_absent():
    assert passed_from_report("Num Total Tests : 5\n") is None


def test_build_step_records_threads_prev_state():
    io = (
        "\n### Environment Interaction 1\n---\n```python\n"
        "a = apis.phone.show_alarms(access_token='t')\n```\n\n```\n"
        '[{"alarm_id": 1, "label": "X", "time": "08:00", "snooze_minutes": 5}]\n```\n'
        "\n### Environment Interaction 2\n---\n```python\nx = 1 + 1\n```\n\n```\n2\n```\n"
    )
    records = build_step_records(io)
    assert records[0]["ui_state"]["kind"] == "phone_alarms"
    assert records[1]["ui_state"]["kind"] == "phone_alarms"  # inherited prev
    assert records[1]["code"] == "x = 1 + 1"
