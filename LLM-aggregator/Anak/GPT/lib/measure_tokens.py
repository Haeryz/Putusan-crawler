#!/usr/bin/env python3
"""Parse Codex stderr logs and report tokens-used per session.

Codex CLI prints a `tokens used\\n<number>` footer to stderr. This scans the
span-extraction logs (codex-span-*.stderr.log and validate-run*.log) and the
legacy logs (codex-target-*.stderr.log), reports per-log usage, and compares
the span-mode average against the recorded legacy baseline.

Usage:
  python lib/measure_tokens.py [logs_dir]
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

LEGACY_BASELINE = 104183  # measured legacy avg (reports/benchmark-*.json)

# Logs may be UTF-8 or UTF-16 (PowerShell Tee-Object). Match digits-with-commas
# after a "tokens used" marker, tolerating interleaved NUL/space bytes.
_TOKENS_RE = re.compile(r"t\W*o\W*k\W*e\W*n\W*s\W*\W*u\W*s\W*e\W*d\W*?([\d,]{2,})", re.IGNORECASE)


def read_loose(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8", "utf-16", "latin-1"):
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1", errors="ignore")


def tokens_in(path: Path) -> int | None:
    text = read_loose(path)
    matches = _TOKENS_RE.findall(text)
    if not matches:
        return None
    return int(matches[-1].replace(",", ""))


def main(argv: list[str]) -> int:
    logs_dir = Path(argv[1]) if len(argv) > 1 else Path(__file__).resolve().parent.parent / "logs"
    span_logs = sorted(
        list(logs_dir.glob("codex-span-*.stderr.log"))
        + list(logs_dir.glob("validate-run*.log"))
    )
    rows = []
    for log in span_logs:
        t = tokens_in(log)
        if t is not None:
            rows.append((log.name, t))
    if not rows:
        print(f"No span-mode token footers found in {logs_dir}")
        return 1
    print(f"{'log':48}{'tokens':>10}")
    for name, t in rows:
        print(f"{name[:46]:48}{t:>10,}")
    avg = sum(t for _, t in rows) / len(rows)
    print("-" * 58)
    print(f"span sessions          : {len(rows)}")
    print(f"span avg tokens/doc    : {avg:,.0f}")
    print(f"legacy baseline avg    : {LEGACY_BASELINE:,}")
    print(f"reduction vs baseline  : {100 * (1 - avg / LEGACY_BASELINE):.0f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
