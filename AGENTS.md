# Repository Guidelines

## Project Structure & Module Organization

Sinergi is a Python 3.12 crawler for downloading public Putusan MA PDFs from `putusan3.mahkamahagung.go.id`. Core crawler code lives in `crawler/`: `cli.py` defines the command-line interface, `crawler.py` contains browser/download orchestration, `parsing.py` handles HTML parsing, and `storage.py` manages JSONL state and deduplication. Root-level tests live in `tests/` and mirror these modules with `test_*.py` files. Runtime artifacts are written to `.browser-profile/` and `downloads/`; both are ignored and should not be committed. `hermes-agent/` is a separate nested project with its own tooling and documentation.

## Build, Test, and Development Commands

- `uv sync`: install project and development dependencies from `pyproject.toml` and `uv.lock`.
- `uv run playwright install chromium`: install the browser used by Playwright-backed crawling.
- `uv run pytest`: run the root Sinergi test suite.
- `uv run python main.py crawl --target-downloads 10`: run the crawler through the local entry point.
- `uv run sinergi crawl --target-downloads 10`: packaged CLI form, when the project script entry point is installed and valid.

Use README examples for crawl options such as `--detail-url`, `--detail-file`, `--parallel-downloads`, and `--no-refresh-profile-snapshot`.

## Coding Style & Naming Conventions

Use idiomatic Python with 4-space indentation, type hints for public functions and structured data, and `pathlib.Path` for filesystem paths. Keep modules focused: parsing logic belongs in `crawler/parsing.py`, persistence in `crawler/storage.py`, and browser workflow changes in `crawler/crawler.py`. Name tests and helper functions descriptively, for example `test_parse_listing_extracts_case_links_and_next`.

## Testing Guidelines

Tests use `pytest`. Add focused unit tests for parsing, retry behavior, inventory counting, and storage changes before touching live crawl behavior. Prefer deterministic HTML snippets and temporary directories over network calls. Run `uv run pytest` before opening a PR; for targeted work, run a single file such as `uv run pytest tests/test_parsing.py`.

## Commit & Pull Request Guidelines

Recent commits use short imperative or descriptive summaries, for example `Enhance crawling logic and add pagination handling in parsing functions`. Keep commit subjects specific to the change and avoid placeholder messages. Pull requests should include a concise description, commands run, linked issue or crawl scenario, and screenshots or log excerpts only when browser behavior or Cloudflare clearance flow changes.

## Security & Configuration Tips

Do not commit `.browser-profile/`, downloaded PDFs, JSONL run state, or logs. The crawler should continue to reject non-`putusan3.mahkamahagung.go.id` download URLs. Treat copied Chrome profiles as local sensitive data and keep secrets or personal browsing state out of fixtures and examples.
