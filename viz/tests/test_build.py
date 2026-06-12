from viz.build_trajectory_json import passed_from_report, build_step_records


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
