from __future__ import annotations

from pathlib import Path


SETUP_SCRIPT = (
    Path(__file__).resolve().parents[1] / "trainer" / "sft" / "setup_runpod.sh"
)


def test_runpod_setup_omits_unused_audio_compiler_stack() -> None:
    script = SETUP_SCRIPT.read_text(encoding="utf-8")
    install_lines = [
        line for line in script.splitlines() if line.startswith("uv pip install")
    ]

    assert all("librosa" not in line for line in install_lines)
    assert "text-only for every profile" in script
