# Sinergi

Browser crawler for downloading public Putusan MA PDF files from
`putusan3.mahkamahagung.go.id`.

## Initial Setup Guide

A new machine, including a MacBook or other Apple device, only needs Git and a
package manager before cloning. The bootstrap script installs everything else
and can run the TPPO and Anak extractors. The raw-text inputs, extraction
progress, and outputs are tracked in the repo, so a fresh clone resumes the run
automatically.

### 1. Prerequisites

- Git, to clone the repo.
- A package manager the bootstrap can drive:
  - macOS: Homebrew (`brew`)
  - Linux: `apt` or `dnf`
  - Windows: `winget`

### 2. Clone

```bash
git clone https://github.com/Haeryz/Putusan-crawler.git
cd Putusan-crawler
```

### 3. Run Setup

macOS/Linux:

```bash
chmod +x setup.sh
./setup.sh            # install prerequisites, then run until usage guard stops
./setup.sh 20         # run 20 sources per corpus for a bounded test
./setup.sh --status   # show pending/completed counts only
```

Windows:

```powershell
.\setup.cmd
.\setup.cmd 20
powershell -ExecutionPolicy Bypass -File setup.ps1 -StatusOnly
```

The macOS/Linux setup path is native Unix. It uses `setup.sh` and the Python
orchestrator `run_extractions.py`; it does not require PowerShell. If Codex is
not logged in, setup launches `codex login`, which opens a browser. Sign in
once; the session is cached in `~/.codex`.

### 4. Run TPPO or Anak Independently

After setup, run both corpora:

```bash
python3 run_extractions.py
python3 run_extractions.py --target 1
python3 run_extractions.py --disable-usage-guard --jobs 4 --target 8
```

Run TPPO only:

```bash
python3 run_extractions.py --corpus TPPO
python3 run_extractions.py --corpus TPPO --status
```

Run Anak only:

```bash
python3 run_extractions.py --corpus Anak
python3 run_extractions.py --corpus Anak --status
```

Useful controls:

```bash
python3 run_extractions.py --model gpt-5-codex
python3 run_extractions.py --reasoning-effort low
python3 run_extractions.py --keep-mcp
```

By default, the extractor processes all pending sources one at a time until the
5h usage guard stops it, a failure occurs, or the corpus is complete. `--target
N` limits the run to N pending sources per selected corpus. `--jobs N` runs up
to N Codex sessions in parallel only when `--disable-usage-guard` is also set.
The usage guard parses Codex `/status`-style text when available and also has a
270-minute wall-clock fallback (`--max-run-minutes`) so AFK runs stop before the
five-hour window is likely exhausted.

> Source PDFs and crawler run logs under `downloads/` are not committed because
> they are large and the extractors do not read them. The raw-text inputs are
> tracked. To re-crawl PDFs and regenerate raw text, see **Crawl** below.

## Setup For Crawler Development

```bash
uv sync
uv run playwright install chromium
```

## Crawl

```bash
uv run sinergi crawl --target-downloads 10
```

The default backend is `managed-chrome`, which copies your selected installed
Chrome profile into `.browser-profile/managed-chrome`, opens real visible
Chrome, waits for Cloudflare to clear, parses Putusan detail pages, and
downloads the site's own `/direktori/download_file/.../pdf/...` links. The
default profile is `Profile 4`.

If Cloudflare shows human verification, complete it in the opened Chrome
window. The crawler will continue automatically after the page loads. The
crawler only accepts URLs from `putusan3.mahkamahagung.go.id`; alternative
download sites are rejected.

Downloaded PDFs are saved to `downloads/pdfs/`. Successful records are written
to `downloads/downloaded.jsonl`; skipped pages and exhausted failures are
written to `downloads/skipped.jsonl`; retry details go to `downloads/run.log`.
Listing crawls skip already downloaded detail pages, while explicit
`--detail-url` and `--detail-file` inputs are always processed again.

Useful options:

```bash
uv run sinergi crawl --target-downloads 10 --out-dir downloads --profile-dir .browser-profile
uv run sinergi crawl --target-downloads 10 --max-candidates 200
uv run sinergi crawl --target-downloads 1 --timeout-seconds 180
uv run sinergi crawl --detail-url https://putusan3.mahkamahagung.go.id/direktori/putusan/zaf14cd81bd6894491e8303832343038.html --no-listing --target-downloads 1
uv run sinergi crawl --detail-file detail-urls.txt --no-listing --target-downloads 10
uv run sinergi crawl --chrome-profile "Profile 4" --target-downloads 10
uv run sinergi crawl --manual-clearance-timeout-seconds 600 --target-downloads 10
uv run sinergi crawl --no-refresh-profile-snapshot --target-downloads 10
uv run sinergi crawl --target-downloads 100 --parallel-downloads 16 --no-refresh-profile-snapshot
uv run sinergi crawl --target-downloads 100 --parallel-downloads 32 --fast-fetch-timeout-seconds 10 --no-refresh-profile-snapshot
uv run sinergi crawl --chrome-version-main 148 --target-downloads 10
uv run sinergi crawl --browser-backend playwright --browser-channel chrome --target-downloads 10
uv run sinergi crawl --target-downloads 10 --restart-listing
uv run sinergi crawl --target-downloads 10 --new-target
```

Listing pagination is resumable. The crawler stores the interrupted listing
page in `<out-dir>/crawl-state.json` and resumes there on the next run, while
still skipping case URLs already recorded in `downloaded.jsonl`. Use
`--restart-listing` only when you intentionally want to discard that pagination
checkpoint and scan from the configured start URL.

Numeric download targets are also resumable. If a target of 264 is interrupted
after 123 verified downloads, running the same target again continues with the
remaining 141. Once the target is completed, a later invocation starts a fresh
target. Use `--new-target` to intentionally discard an unfinished target and
start counting from the current verified total.
