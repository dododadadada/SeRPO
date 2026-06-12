from pathlib import Path
from viz.parse_io import parse_io, Step

FIXTURE = Path(__file__).parent / "fixtures" / "sample_io.md"


def test_parses_three_steps():
    steps = parse_io(FIXTURE.read_text())
    assert len(steps) == 3
    assert all(isinstance(s, Step) for s in steps)


def test_step_numbers_are_sequential():
    steps = parse_io(FIXTURE.read_text())
    assert [s.step for s in steps] == [1, 2, 3]


def test_code_and_output_split():
    steps = parse_io(FIXTURE.read_text())
    assert steps[1].code == "login_result = apis.venmo.login(username='a@b.com', password='x')\nprint(login_result)"
    assert steps[1].output == '{"access_token": "TOK"}'


def test_classifies_app_and_api():
    steps = parse_io(FIXTURE.read_text())
    assert steps[0].app == "api_docs"
    assert steps[0].api == "show_api_descriptions"
    assert steps[1].app == "venmo"
    assert steps[1].api == "login"
    assert steps[2].app == "supervisor"
    assert steps[2].api == "complete_task"


def test_empty_code_step_has_none_app():
    md = "\n### Environment Interaction 1\n---\n```python\nx = 1 + 1\n```\n\n```\n2\n```\n"
    steps = parse_io(md)
    assert steps[0].app is None
    assert steps[0].api is None


def test_step_without_output_fence_is_skipped():
    md = "\n### Environment Interaction 1\n---\n```python\nx=1\n```\n\n### Environment Interaction 2\n---\n```python\ny=2\n```\n\n```\nok\n```\n"
    steps = parse_io(md)
    assert len(steps) == 1
    assert steps[0].step == 2
