#!/usr/bin/env python3
"""
Native, cross-platform orchestrator for the TPPO + Anak Codex span extraction.

This replaces the PowerShell launchers. It needs only Python 3 (preinstalled on
macOS) and the Codex CLI -- no PowerShell. The actual extraction logic still
lives in the per-corpus lib scripts (court_extract.py / anak_extract.py); this
file is just the glue: find pending sources, clean them, ask Codex for spans,
expand the spans into schema-conforming JSON, and append a checkpoint.

The on-disk format (output/*.json + progress.jsonl) is identical to the
PowerShell path, so runs started on Windows resume here and vice versa.

Usage:
    python3 run_extractions.py                 # 1 source per corpus
    python3 run_extractions.py --target 20     # 20 per corpus
    python3 run_extractions.py --status        # show pending/completed counts
    python3 run_extractions.py --corpus TPPO   # one corpus only
    python3 run_extractions.py --jobs 4        # run up to 4 Codex sessions at once
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent

CORPORA: dict[str, dict] = {
    "TPPO": {
        "label": "TPPO",
        "guide": "TPPO Format PDF",
        "input_dir": "downloads/TPPO/raw-text",
        "out_dir": "LLM-aggregator/TPPO/GPT/output",
        "logs_dir": "LLM-aggregator/TPPO/GPT/logs",
        "spans_dir": "LLM-aggregator/TPPO/GPT/.spans",
        "progress": "LLM-aggregator/TPPO/GPT/progress.jsonl",
        "instruction": "LLM-aggregator/TPPO/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md",
        "spec": "LLM-aggregator/TPPO/GPT/SPAN_EXTRACTION_SPEC.md",
        "lib": "LLM-aggregator/TPPO/GPT/lib/court_extract.py",
    },
    "Anak": {
        "label": "Anak",
        "guide": "SKKMA PDF",
        "input_dir": "downloads/kasus anak/raw-text",
        "out_dir": "LLM-aggregator/Anak/GPT/output",
        "logs_dir": "LLM-aggregator/Anak/GPT/logs",
        "spans_dir": "LLM-aggregator/Anak/GPT/.spans",
        "progress": "LLM-aggregator/Anak/GPT/progress.jsonl",
        "instruction": "LLM-aggregator/Anak/GPT/CODEX_EXTRACTION_INSTRUCTIONS.md",
        "spec": "LLM-aggregator/Anak/GPT/SPAN_EXTRACTION_SPEC.md",
        "lib": "LLM-aggregator/Anak/GPT/lib/anak_extract.py",
    },
}


# --------------------------------------------------------------------------- #
# Codex helpers
# --------------------------------------------------------------------------- #
def resolve_codex() -> str:
    exe = shutil.which("codex")
    if not exe:
        sys.exit("Codex CLI was not found on PATH. Install it and run 'codex login', then retry.")
    return exe


def init_codex_no_mcp_home() -> Path | None:
    """Build a CODEX_HOME with all MCP/plugin/marketplace tables stripped.

    Mirrors lib/codex-no-mcp.ps1: extraction needs no MCP servers, but Codex
    loads every one from the user config on each `codex exec` (failing-auth
    spam, startup latency, wasted tool-schema tokens). We copy the user config
    with those tables removed, plus auth.json, into a private home and point the
    Codex child at it -- never mutating the real config.
    """
    real_home = Path(os.environ.get("CODEX_HOME") or (Path.home() / ".codex"))
    config_src = real_home / "config.toml"
    if not config_src.exists():
        print(f"[warn] Codex config.toml not found at {config_src}; running with MCP enabled.")
        return None

    if os.name == "nt" and os.environ.get("LOCALAPPDATA"):
        data_root = Path(os.environ["LOCALAPPDATA"])
    elif os.environ.get("XDG_CACHE_HOME"):
        data_root = Path(os.environ["XDG_CACHE_HOME"])
    else:
        data_root = Path.home() / ".cache"
    alt_home = data_root / "sinergi-codex-nomcp"
    alt_home.mkdir(parents=True, exist_ok=True)

    drop = re.compile(r"^\s*\[(mcp_servers|plugins|marketplaces)")
    header = re.compile(r"^\s*\[")
    kept: list[str] = []
    skip = False
    for line in config_src.read_text(encoding="utf-8", errors="replace").splitlines():
        if header.match(line):
            skip = bool(drop.match(line))
        if not skip:
            kept.append(line)
    (alt_home / "config.toml").write_text("\n".join(kept) + "\n", encoding="utf-8")

    alt_plugins = alt_home / "plugins"
    if alt_plugins.exists():
        shutil.rmtree(alt_plugins, ignore_errors=True)

    for name in ("auth.json", "installation_id", "version.json"):
        src = real_home / name
        if src.exists():
            shutil.copy2(src, alt_home / name)

    if not (alt_home / "auth.json").exists():
        print(f"[warn] auth.json not found in {real_home}; MCP-disabled home may not be authenticated.")
    return alt_home


def codex_args(model: str, reasoning: str, json_events: bool) -> list[str]:
    args = ["exec", "--cd", str(REPO_ROOT), "--sandbox", "workspace-write"]
    if model.strip():
        args += ["--model", model]
    if reasoning.strip():
        # Lower reasoning effort is a legitimate token lever for this bounded,
        # deterministically-verified task.
        args += ["-c", f'model_reasoning_effort="{reasoning}"']
    if json_events:
        args += ["--json"]
    args += ["-"]
    return args


# --------------------------------------------------------------------------- #
# Queue discovery
# --------------------------------------------------------------------------- #
def completed_set(progress_path: Path) -> set[str]:
    done: set[str] = set()
    if not progress_path.exists():
        return done
    for line in progress_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except Exception:
            continue  # ignore malformed lines during queue discovery
        if rec.get("status") == "completed" and rec.get("source_file"):
            done.add(rec["source_file"])
    return done


def pending_sources(cfg: dict) -> list[Path]:
    in_dir = REPO_ROOT / cfg["input_dir"]
    if not in_dir.exists():
        return []
    done = completed_set(REPO_ROOT / cfg["progress"])
    out_dir = REPO_ROOT / cfg["out_dir"]
    outs = {p.stem for p in out_dir.glob("*.json")} if out_dir.exists() else set()
    return [
        p for p in sorted(in_dir.glob("*.txt"), key=lambda x: x.name)
        if p.name not in done and p.stem not in outs
    ]


# --------------------------------------------------------------------------- #
# Python lib calls
# --------------------------------------------------------------------------- #
def run_lib(cfg: dict, lib_args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(REPO_ROOT / cfg["lib"]), *lib_args],
        capture_output=True, text=True, encoding="utf-8",
    )


def numbered_source(cfg: dict, source: Path) -> str:
    r = run_lib(cfg, ["clean", str(source)])
    if r.returncode != 0:
        raise RuntimeError(f"clean failed for {source.name}: {r.stderr.strip()}")
    return r.stdout.rstrip("\n")


def expand_spans(cfg: dict, source: Path, spans_path: Path, out_path: Path, source_name: str) -> str:
    if not spans_path.exists():
        raise RuntimeError(f"Spans file was not produced by Codex: {spans_path}")
    r = run_lib(cfg, [
        "expand",
        "--source", str(source),
        "--spans", str(spans_path),
        "--out", str(out_path),
        "--source-file", source_name,
        "--source-path", f'{cfg["input_dir"]}/{source_name}',
    ])
    if r.returncode != 0:
        raise RuntimeError(f"Span expansion failed for {source_name}: {r.stderr.strip()}")
    lines = [ln for ln in r.stdout.splitlines() if ln.strip()]
    if not lines:
        raise RuntimeError(f"Span expansion produced no summary for {source_name}.")
    return lines[-1]


def add_checkpoint(progress_path: Path, summary: str) -> None:
    with open(progress_path, "a", encoding="utf-8") as f:
        f.write(summary.rstrip("\n") + "\n")


# --------------------------------------------------------------------------- #
# Prompt
# --------------------------------------------------------------------------- #
def build_prompt(cfg: dict, source_name: str, spans_forward: str, numbered: str, pause: str) -> str:
    spec = (REPO_ROOT / cfg["spec"]).read_text(encoding="utf-8")
    return f"""You are Codex running the token-optimized {cfg['label']} span-extraction task in:
{REPO_ROOT}

Assigned source: {cfg['input_dir']}/{source_name}
The cleaned, line-numbered source is provided INLINE below. Do NOT open the
source file, the {cfg['guide']}, {cfg['instruction']}, or any other guide --
everything you need is inline. Do not re-read or search files.

YOUR ONLY OUTPUT: write a spans JSON file to exactly this path and nothing else:
  {spans_forward}
Do NOT write the final output JSON. Do NOT edit {cfg['progress']}. A deterministic
post-processor expands your spans into the schema-conforming artifact and the
checkpoint. After writing the spans file, stop.

{spec}

=== CLEANED LINE-NUMBERED SOURCE (1-based; point your line ranges into these) ===
{numbered}
=== END SOURCE ===

Work in a single pass: do not re-read or re-verify files. Write the spans JSON
to {spans_forward} covering all 31 section keys, then stop. {pause}"""


# --------------------------------------------------------------------------- #
# Per-source work
# --------------------------------------------------------------------------- #
def prepare(cfg: dict, source: Path, pause: str) -> dict:
    """Clean the source and build everything the Codex call needs."""
    numbered = numbered_source(cfg, source)
    spans_path = REPO_ROOT / cfg["spans_dir"] / (source.stem + ".spans.json")
    spans_path.parent.mkdir(parents=True, exist_ok=True)
    if spans_path.exists():
        spans_path.unlink()
    spans_forward = f'{cfg["spans_dir"]}/{source.stem}.spans.json'
    return {
        "source": source,
        "spans_path": spans_path,
        "out_path": REPO_ROOT / cfg["out_dir"] / (source.stem + ".json"),
        "prompt": build_prompt(cfg, source.name, spans_forward, numbered, pause),
    }


def codex_live(codex: str, args: list[str], env: dict, prompt: str) -> int:
    # Inherit stdout/stderr so Codex progress is visible in the terminal.
    return subprocess.run([codex, *args], input=prompt, text=True, encoding="utf-8", env=env).returncode


def codex_logged(codex: str, args: list[str], env: dict, prompt: str, out_log: Path, err_log: Path) -> int:
    with open(out_log, "w", encoding="utf-8") as so, open(err_log, "w", encoding="utf-8") as se:
        return subprocess.run(
            [codex, *args], input=prompt, text=True, encoding="utf-8", env=env, stdout=so, stderr=se
        ).returncode


# --------------------------------------------------------------------------- #
# Corpus runner
# --------------------------------------------------------------------------- #
def show_status(cfg: dict) -> None:
    in_dir = REPO_ROOT / cfg["input_dir"]
    out_dir = REPO_ROOT / cfg["out_dir"]
    sources = len(list(in_dir.glob("*.txt"))) if in_dir.exists() else 0
    completed = len(completed_set(REPO_ROOT / cfg["progress"]))
    outputs = len(list(out_dir.glob("*.json"))) if out_dir.exists() else 0
    pending = len(pending_sources(cfg))
    print(f"  Sources:  {sources}")
    print(f"  Progress: {completed} completed checkpoint record(s)")
    print(f"  Outputs:  {outputs} JSON file(s)")
    print(f"  Pending:  {pending} source file(s)")


def run_corpus(cfg: dict, codex: str, args: list[str], env: dict, target: int,
               pause: str, jobs: int) -> int:
    for sub in ("out_dir", "logs_dir", "spans_dir"):
        (REPO_ROOT / cfg[sub]).mkdir(parents=True, exist_ok=True)
    progress_path = REPO_ROOT / cfg["progress"]
    progress_path.touch(exist_ok=True)

    pending = pending_sources(cfg)[:target]
    if not pending:
        print("  No pending sources.")
        return 0
    if len(pending) < target:
        print(f"  Only {len(pending)} pending source(s) available; reducing target.")

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    failures = 0

    if jobs > 1 and len(pending) > 1:
        print(f"  Execution: parallel (up to {jobs} Codex sessions)")
        prepared = [prepare(cfg, s, pause) for s in pending]
        logs_dir = REPO_ROOT / cfg["logs_dir"]

        def launch(item: tuple[int, dict]) -> tuple[dict, int]:
            i, p = item
            out_log = logs_dir / f"codex-span-{stamp}-{i}.stdout.log"
            err_log = logs_dir / f"codex-span-{stamp}-{i}.stderr.log"
            print(f"  Starting session {i}/{len(prepared)}: {p['source'].name}")
            rc = codex_logged(codex, args, env, p["prompt"], out_log, err_log)
            return p, rc

        with ThreadPoolExecutor(max_workers=jobs) as pool:
            results = list(pool.map(launch, enumerate(prepared, 1)))

        for p, rc in results:
            name = p["source"].name
            if rc != 0:
                print(f"  [fail] Codex session for {name} exited {rc} (see logs/).")
                failures += 1
                continue
            try:
                summary = expand_spans(cfg, p["source"], p["spans_path"], p["out_path"], name)
                add_checkpoint(progress_path, summary)
                print(f"  Completed: {name} -> {p['out_path']}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  [fail] {name}: {exc}")
                failures += 1
    else:
        print(f"  Execution: sequential ({len(pending)} Codex session(s))")
        for i, source in enumerate(pending, 1):
            print(f"\n  Session {i} of {len(pending)}: {source.name}")
            try:
                p = prepare(cfg, source, pause)
                rc = codex_live(codex, args, env, p["prompt"])
                if rc != 0:
                    print(f"  [fail] Codex exited {rc} for {source.name}.")
                    failures += 1
                    continue
                summary = expand_spans(cfg, source, p["spans_path"], p["out_path"], source.name)
                add_checkpoint(progress_path, summary)
                print(f"  Completed: {source.name} -> {p['out_path']}")
            except Exception as exc:  # noqa: BLE001 - report and continue
                print(f"  [fail] {source.name}: {exc}")
                failures += 1

    return 1 if failures else 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Run the TPPO + Anak Codex span extractors (native Python).")
    ap.add_argument("--target", type=int, default=1, help="Sources to process per corpus (default 1).")
    ap.add_argument("--corpus", choices=["TPPO", "Anak", "both"], default="both")
    ap.add_argument("--model", default="", help="Override the Codex model.")
    ap.add_argument("--reasoning-effort", default="low", help='Codex model_reasoning_effort (default "low"; "" for model default).')
    ap.add_argument("--jobs", type=int, default=1, help="Max parallel Codex sessions per corpus (default 1).")
    ap.add_argument("--no-pause", action="store_true", help="Tell Codex never to pause for confirmation.")
    ap.add_argument("--json-events", action="store_true", help="Pass --json to Codex.")
    ap.add_argument("--keep-mcp", action="store_true", help="Do not build the MCP-disabled Codex home.")
    ap.add_argument("--status", action="store_true", help="Show counts for each corpus and exit.")
    a = ap.parse_args(argv)

    names = ["TPPO", "Anak"] if a.corpus == "both" else [a.corpus]

    if a.status:
        for name in names:
            print(f"=== {name} ===")
            show_status(CORPORA[name])
        return 0

    if a.target < 1:
        sys.exit("--target must be >= 1.")

    codex = resolve_codex()
    args = codex_args(a.model, a.reasoning_effort, a.json_events)

    env = os.environ.copy()
    if not a.keep_mcp:
        alt_home = init_codex_no_mcp_home()
        if alt_home:
            env["CODEX_HOME"] = str(alt_home)
            print(f"MCP: disabled (CODEX_HOME={alt_home})")

    pause = (
        "Do not pause for user confirmation. Make reasonable assumptions and keep the extraction loop moving."
        if a.no_pause else
        "Do not pause for user confirmation unless the next action would be destructive outside the GPT extraction paths."
    )

    overall = 0
    for name in names:
        cfg = CORPORA[name]
        print("\n" + "=" * 67)
        print(f" {name} extractor")
        print("=" * 67)
        rc = run_corpus(cfg, codex, args, env, a.target, pause, a.jobs)
        if rc != 0:
            overall = 1

    print()
    if overall:
        print("Completed with failures. See the messages/logs above.")
    else:
        print("Both extractors finished." if len(names) > 1 else f"{names[0]} extractor finished.")
    return overall


if __name__ == "__main__":
    raise SystemExit(main())
