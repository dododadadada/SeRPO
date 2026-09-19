"""Idempotent source patches for the editable appworld-agents install
(appworld/experiments/, gitignored). Run by scripts/launch_vllm.sh; safe to re-run.

1. react_code_agent.py: vLLM >= 0.29 returns `reasoning_content: None` explicitly
   in chat messages; upstream does `.get("reasoning_content", "").strip()` and
   crashes on the first step. Treat None as "".
"""
from __future__ import annotations
import sys
from pathlib import Path

PATCHES = [
    (Path("appworld/experiments/code/simplified/react_code_agent.py"),
     'reasoning_content = output.get("reasoning_content", "").strip()',
     'reasoning_content = (output.get("reasoning_content") or "").strip()'),
]


def apply(root: Path = Path(".")) -> int:
    changed = 0
    for rel, old, new in PATCHES:
        p = root / rel
        if not p.exists():
            print(f"[appworld_patches] missing {p} — is appworld-agents installed under appworld/experiments?", file=sys.stderr)
            return 1
        s = p.read_text()
        if new in s:
            continue
        if old not in s:
            print(f"[appworld_patches] anchor not found in {p}: {old!r}", file=sys.stderr)
            return 1
        p.write_text(s.replace(old, new))
        changed += 1
        print(f"[appworld_patches] patched {p}")
    print(f"[appworld_patches] ok ({changed} file(s) changed)")
    return 0


if __name__ == "__main__":
    sys.exit(apply())
