# Sinergi

Browser crawler for downloading public Putusan MA PDF files from
`putusan3.mahkamahagung.go.id`.

## Setup

```powershell
uv sync
uv run playwright install chromium
```

## Crawl

```powershell
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
to `downloads/downloaded.jsonl`; skipped pages and exhausted failures are written
to `downloads/skipped.jsonl`; retry details go to `downloads/run.log`. Listing
crawls skip already downloaded detail pages, while explicit `--detail-url` and
`--detail-file` inputs are always processed again.

Useful options:

```powershell
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

Listing pagination is resumable. The crawler stores the interrupted listing page in
`<out-dir>/crawl-state.json` and resumes there on the next run, while still skipping
case URLs already recorded in `downloaded.jsonl`. Use `--restart-listing` only when
you intentionally want to discard that pagination checkpoint and scan from the
configured start URL.

Numeric download targets are also resumable. If a target of 264 is interrupted
after 123 verified downloads, running the same target again continues with the
remaining 141. Once the target is completed, a later invocation starts a fresh
target. Use `--new-target` to intentionally discard an unfinished target and
start counting from the current verified total.
