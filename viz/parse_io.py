"""Parse AppWorld environment_io.md transcripts into structured steps."""
import re
from dataclasses import dataclass


@dataclass
class Step:
    step: int
    code: str
    output: str
    app: str | None      # e.g. "venmo", or None if the code calls no apis.* method
    api: str | None       # e.g. "login"


# Split the full document into per-block chunks at each "### Environment Interaction N" header.
_HEADER_RE = re.compile(r"### Environment Interaction (\d+)\s*\n")

# Within a single block, extract the code fence and output fence.
# Non-greedy (.*?) stops at the first closing ```. AppWorld outputs never
# embed triple backticks, so this is safe for the real format.
_CODE_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
_OUT_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)

# Last apis.<app>.<api>( call in the code is treated as the primary/state-changing one.
_API_RE = re.compile(r"apis\.([a-z_][a-z0-9_]*)\.([a-z_][a-z0-9_]*)\s*\(")


def _classify(code: str) -> tuple[str | None, str | None]:
    matches = _API_RE.findall(code)
    if not matches:
        return None, None
    app, api = matches[-1]
    return app, api


def parse_io(text: str) -> list[Step]:
    # Split document at each block header, keeping the number captured.
    parts = _HEADER_RE.split(text)
    # parts[0] is text before the first header (usually blank).
    # Then alternating: num_str, block_text, num_str, block_text, ...
    steps: list[Step] = []
    i = 1
    while i + 1 <= len(parts) - 1:
        num = int(parts[i])
        block = parts[i + 1]
        i += 2

        code_m = _CODE_RE.search(block)
        if code_m is None:
            continue
        code = code_m.group(1).strip("\n")

        # Output fence is the first plain ``` fence AFTER the python code fence.
        remaining = block[code_m.end():]
        out_m = _OUT_RE.search(remaining)
        if out_m is None:
            continue
        output = out_m.group(1).strip("\n")

        app, api = _classify(code)
        steps.append(Step(step=num, code=code, output=output, app=app, api=api))
    return steps
