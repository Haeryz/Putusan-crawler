from __future__ import annotations

import json
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from urllib.parse import unquote


@dataclass(frozen=True)
class CrawlRecord:
    status: str
    detail_url: str
    pdf_url: str | None = None
    output_path: str | None = None
    title: str | None = None
    filename: str | None = None
    error: str | None = None
    timestamp: str = ""

    def to_json(self) -> str:
        data = asdict(self)
        if not data["timestamp"]:
            data["timestamp"] = datetime.now(UTC).isoformat()
        return json.dumps(data, ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class DeduplicationSummary:
    kept_records: int = 0
    duplicate_records: int = 0
    invalid_records: int = 0
    moved_files: int = 0


@dataclass(frozen=True)
class TargetProgress:
    target: int
    completed: int
    baseline_downloaded: int
    resumed: bool


class JsonlStore:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.success_path = out_dir / "downloaded.jsonl"
        self.skipped_path = out_dir / "skipped.jsonl"
        self.log_path = out_dir / "run.log"
        self.state_path = out_dir / "crawl-state.json"

    def prepare(self) -> None:
        self.out_dir.mkdir(parents=True, exist_ok=True)
        (self.out_dir / "pdfs").mkdir(parents=True, exist_ok=True)
        summary = self.deduplicate_downloads()
        if summary.duplicate_records or summary.invalid_records or summary.moved_files:
            self.log(
                "dedupe "
                f"kept={summary.kept_records} "
                f"duplicates={summary.duplicate_records} "
                f"invalid={summary.invalid_records} "
                f"moved_files={summary.moved_files}"
            )

    def downloaded_detail_urls(self) -> set[str]:
        urls: set[str] = set()
        if not self.success_path.exists():
            return urls

        for line in self.success_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if record.get("status") == "downloaded" and record.get("detail_url"):
                urls.add(str(record["detail_url"]))
        return urls

    def excluded_detail_urls(self) -> set[str]:
        urls: set[str] = set()
        if not self.skipped_path.exists():
            return urls
        for line in self.skipped_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if (
                record.get("status") == "skipped_duplicate_content"
                and record.get("detail_url")
            ):
                urls.add(str(record["detail_url"]))
        return urls

    def processed_detail_urls(self) -> set[str]:
        return self.downloaded_detail_urls() | self.excluded_detail_urls()

    def begin_or_resume_target(
        self,
        target_key: str,
        target: int,
        current_downloaded: int,
        *,
        force_new: bool = False,
    ) -> TargetProgress:
        state = self._read_state()
        targets = state.setdefault("download_targets", {})
        if not isinstance(targets, dict):
            targets = {}
            state["download_targets"] = targets
        active = targets.get(target_key)
        if (
            not force_new
            and isinstance(active, dict)
            and active.get("completed") is not True
            and active.get("target") == target
        ):
            baseline = int(active.get("baseline_downloaded") or 0)
            completed = min(target, max(0, current_downloaded - baseline))
            return TargetProgress(target, completed, baseline, resumed=True)

        targets[target_key] = {
            "target": target,
            "baseline_downloaded": current_downloaded,
            "completed": False,
            "started_at": datetime.now(UTC).isoformat(),
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._write_state(state)
        return TargetProgress(target, 0, current_downloaded, resumed=False)

    def complete_target(self, target_key: str) -> None:
        state = self._read_state()
        targets = state.get("download_targets")
        if not isinstance(targets, dict):
            return
        target = targets.get(target_key)
        if not isinstance(target, dict):
            return
        target["completed"] = True
        target["completed_at"] = datetime.now(UTC).isoformat()
        target["updated_at"] = datetime.now(UTC).isoformat()
        self._write_state(state)

    def append(self, record: CrawlRecord) -> None:
        target = self.success_path if record.status == "downloaded" else self.skipped_path
        with target.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json())
            handle.write("\n")

    def load_listing_checkpoint(self, checkpoint_key: str) -> str | None:
        if not self.state_path.exists():
            return None
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return None
        checkpoints = state.get("listing_checkpoints")
        if not isinstance(checkpoints, dict):
            return None
        checkpoint = checkpoints.get(checkpoint_key)
        if not isinstance(checkpoint, dict):
            return None
        listing_url = checkpoint.get("listing_url")
        return str(listing_url) if listing_url else None

    def has_listing_checkpoint_state(self, checkpoint_key: str) -> bool:
        if not self.state_path.exists():
            return False
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return False
        checkpoints = state.get("listing_checkpoints")
        return isinstance(checkpoints, dict) and checkpoint_key in checkpoints

    def infer_listing_checkpoint_from_log(self, start_url: str) -> str | None:
        if not self.log_path.exists():
            return None
        category_prefix = start_url.removesuffix(".html")
        pattern = re.compile(
            r"(?:managed-listing-clicks|managed-listing-fast|listing|"
            r"playwright-cdp-listing|undetected-listing) "
            r"(?:page=\d+ )?url=(https?://\S+)"
        )
        inferred: str | None = None
        try:
            lines = self.log_path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return None
        for line in lines:
            match = pattern.search(line)
            if not match:
                continue
            listing_url = match.group(1)
            if listing_url == start_url or listing_url.startswith(f"{category_prefix}/"):
                inferred = listing_url
        return inferred

    def save_listing_checkpoint(self, checkpoint_key: str, listing_url: str) -> None:
        state = self._read_state()
        checkpoints = state.setdefault("listing_checkpoints", {})
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            state["listing_checkpoints"] = checkpoints
        checkpoints[checkpoint_key] = {
            "listing_url": listing_url,
            "completed": False,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._write_state(state)

    def clear_listing_checkpoint(self, checkpoint_key: str) -> None:
        state = self._read_state()
        checkpoints = state.get("listing_checkpoints")
        if not isinstance(checkpoints, dict):
            checkpoints = {}
            state["listing_checkpoints"] = checkpoints
        checkpoints[checkpoint_key] = {
            "listing_url": None,
            "completed": True,
            "updated_at": datetime.now(UTC).isoformat(),
        }
        self._write_state(state)

    def deduplicate_downloads(self) -> DeduplicationSummary:
        if not self.success_path.exists():
            return DeduplicationSummary()

        kept_records: list[dict[str, object]] = []
        kept_output_paths: set[Path] = set()
        seen_detail_urls: set[str] = set()
        seen_pdf_urls: set[str] = set()
        seen_hashes: set[str] = set()
        duplicate_records = 0
        invalid_records = 0
        moved_files = 0

        for line in self.success_path.read_text(encoding="utf-8-sig").splitlines():
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                invalid_records += 1
                continue

            if record.get("status") != "downloaded":
                invalid_records += 1
                continue

            output_path = self._record_output_path(record)
            if output_path is None:
                invalid_records += 1
                continue
            try:
                verify_pdf(output_path)
            except ValueError:
                invalid_records += 1
                if self._move_duplicate_file(output_path):
                    moved_files += 1
                continue

            file_hash = _file_sha256(output_path)
            detail_url = str(record.get("detail_url") or "")
            pdf_url = str(record.get("pdf_url") or "")
            duplicate = (
                bool(detail_url and detail_url in seen_detail_urls)
                or bool(pdf_url and pdf_url in seen_pdf_urls)
                or file_hash in seen_hashes
            )
            if duplicate:
                duplicate_records += 1
                resolved = output_path.resolve()
                if resolved not in kept_output_paths and self._move_duplicate_file(output_path):
                    moved_files += 1
                continue

            kept_records.append(record)
            kept_output_paths.add(output_path.resolve())
            if detail_url:
                seen_detail_urls.add(detail_url)
            if pdf_url:
                seen_pdf_urls.add(pdf_url)
            seen_hashes.add(file_hash)

        if duplicate_records or invalid_records:
            temp_path = self.success_path.with_suffix(".jsonl.tmp")
            with temp_path.open("w", encoding="utf-8") as handle:
                for record in kept_records:
                    handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
                    handle.write("\n")
            temp_path.replace(self.success_path)

        return DeduplicationSummary(
            kept_records=len(kept_records),
            duplicate_records=duplicate_records,
            invalid_records=invalid_records,
            moved_files=moved_files,
        )

    def log(self, message: str) -> None:
        timestamp = datetime.now(UTC).isoformat()
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {message}\n")

    def _read_state(self) -> dict[str, object]:
        if not self.state_path.exists():
            return {}
        try:
            state = json.loads(self.state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
        return state if isinstance(state, dict) else {}

    def _write_state(self, state: dict[str, object]) -> None:
        temp_path = self.state_path.with_suffix(".json.tmp")
        temp_path.write_text(
            json.dumps(state, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temp_path.replace(self.state_path)

    def _record_output_path(self, record: dict[str, object]) -> Path | None:
        raw_path = str(record.get("output_path") or "")
        if not raw_path:
            return None

        path = Path(raw_path)
        if path.is_absolute():
            return path
        if path.exists():
            return path
        return self.out_dir / path

    def _move_duplicate_file(self, path: Path) -> bool:
        if not path.exists():
            return False

        duplicate_dir = self.out_dir / "duplicates"
        duplicate_dir.mkdir(parents=True, exist_ok=True)
        target = unique_path(duplicate_dir / path.name)
        shutil.move(str(path), str(target))
        return True


def sanitize_filename(filename: str | None, fallback_stem: str, suffix: str = ".pdf") -> str:
    raw = unquote(filename or "").strip() or f"{fallback_stem}{suffix}"
    raw = raw.replace("\\", "_").replace("/", "_")
    raw = re.sub(r"[\x00-\x1f<>:\"|?*]+", "_", raw)
    raw = re.sub(r"\s+", " ", raw).strip(" .")
    if not raw:
        raw = f"{fallback_stem}{suffix}"
    if not raw.lower().endswith(suffix):
        raw = f"{raw}{suffix}"
    return raw


def unique_path(path: Path) -> Path:
    if not path.exists():
        return path

    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10_000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate

    raise RuntimeError(f"could not create unique filename for {path}")


def verify_pdf(path: Path) -> None:
    if not path.exists():
        raise ValueError(f"downloaded file does not exist: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"downloaded file is empty: {path}")
    with path.open("rb") as handle:
        header = handle.read(5)
    if header != b"%PDF-":
        raise ValueError(f"downloaded file is not a PDF: {path}")


def _file_sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
