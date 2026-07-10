import argparse
import json
import re
from pathlib import Path


SOURCE = Path("notebooks/.plan_c_notebook_source.py")
TARGET = Path("notebooks/Qwen3_5_(4B)-encoder.ipynb")


def source_lines(value: str) -> list[str]:
    return value.splitlines(keepends=True) or [""]


def parse_percent_script(text: str) -> list[dict]:
    cells: list[dict] = []
    cell_type: str | None = None
    body: list[str] = []

    def flush() -> None:
        nonlocal body
        if cell_type is None:
            return
        if cell_type == "markdown":
            markdown = []
            for line in body:
                if line.startswith("# "):
                    markdown.append(line[2:])
                elif line.startswith("#"):
                    markdown.append(line[1:])
                else:
                    markdown.append(line)
            source = "".join(markdown).strip() + "\n"
        else:
            source = "".join(body).strip() + "\n"
        if source.strip():
            cells.append({"cell_type": cell_type, "source": source})
        body = []

    for line in text.splitlines(keepends=True):
        marker = re.fullmatch(r"# %%(\s+\[markdown\])?\s*\r?\n?", line)
        if marker:
            flush()
            cell_type = "markdown" if marker.group(1) else "code"
        else:
            body.append(line)
    flush()
    return cells


def summarize(cells: list[dict]) -> None:
    for index, cell in enumerate(cells):
        source = cell["source"]
        first = next(
            (line.strip() for line in source.splitlines() if line.strip()),
            "<empty>",
        )
        print(
            f"{index:02d} {cell['cell_type']:<8} "
            f"lines={len(source.splitlines()):4d} chars={len(source):6d} :: "
            f"{first[:120]}"
        )


def merge_code(cells: list[dict], *indices: int) -> dict:
    return {
        "cell_type": "code",
        "source": "\n\n".join(cells[index]["source"].strip() for index in indices)
        + "\n",
    }


def markdown(source: str) -> dict:
    return {"cell_type": "markdown", "source": source.strip() + "\n"}


def clean_layout(cells: list[dict]) -> list[dict]:
    if len(cells) != 36:
        raise RuntimeError(f"Expected 36 source cells, found {len(cells)}")
    matrix_definitions, training_call = cells[29]["source"].rsplit(
        "\nEXPERIMENT_RESULTS = run_required_experiments()", 1
    )
    training_engine = merge_code(cells, 17, 19, 21)
    training_engine["source"] = (
        training_engine["source"].rstrip()
        + "\n\n"
        + matrix_definitions.strip()
        + "\n"
    )
    return [
        cells[0],
        markdown(
            "## 1. Install dependencies\n\n"
            "Run once in a fresh Colab runtime. The committed notebook contains "
            "no outputs or credentials."
        ),
        cells[1],
        markdown(
            "## 2. Hyperparameters, runtime, and W&B\n\n"
            "Edit only the TrainConfig values in the next cell. It contains the "
            "model, optimization defaults, experiment matrix, reporting seeds, "
            "resume identifiers, and smoke-test switch."
        ),
        merge_code(cells, 3, 4),
        markdown(
            "## 3. Label-free preprocessing and exact supervision\n\n"
            "Semantic units are constructed from input_text alone. Gold "
            "sections_json spans are aligned and attached only afterward, so "
            "the same preprocessing path works for unseen text without leaking "
            "target boundaries."
        ),
        merge_code(cells, 6, 8),
        markdown(
            "## 4. Qwen encoder and structured heads\n\n"
            "Qwen supplies local semantic representations; the vocabulary head "
            "is never called. A 512-dimensional unit projector, bidirectional "
            "GRU, boundary filter, semi-CRF, and auxiliary heads form the trainable "
            "extractor."
        ),
        merge_code(cells, 10, 11),
        markdown(
            "## 5. Filtered semi-Markov objective and artifact contract\n\n"
            "This section defines exact normalization/Viterbi decoding, the joint "
            "loss, optimizer groups, experiment-isolated W&B checkpoints, best-"
            "weight restoration, and resumable state."
        ),
        merge_code(cells, 13, 15),
        markdown(
            "## 6. Training engine\n\n"
            "Stage 1 uses document-balanced local training, Stage 2 restores or "
            "builds the versioned embedding cache, and Stage 3 performs gradient-"
            "cached joint tuning. The required systems, component ablations, and "
            "three reporting seeds are orchestrated here."
        ),
        training_engine,
        markdown(
            "## 7. Decoding, metrics, and mandatory baselines\n\n"
            "A0 position-only and A1 whitespace-only are evaluated beside learned "
            "systems. Metrics are source-balanced and include measured promotion "
            "gates, JSON validity, verbatim slicing, filter recall, and resource use."
        ),
        merge_code(cells, 23, 24, 25, 27),
        cells[28],
        {
            "cell_type": "code",
            "source": (
                "# Run the complete configured training matrix.\n"
                "EXPERIMENT_RESULTS = run_required_experiments()\n"
            ),
        },
        cells[30],
        merge_code(cells, 31, 32),
        cells[33],
        merge_code(cells, 34, 35),
    ]


def write_notebook(cells: list[dict]) -> None:
    metadata = {}
    if TARGET.exists():
        metadata = json.loads(TARGET.read_text(encoding="utf-8")).get("metadata", {})
    metadata.setdefault(
        "kernelspec",
        {"display_name": "Python 3", "language": "python", "name": "python3"},
    )
    metadata.setdefault(
        "language_info",
        {"name": "python", "version": "3.12"},
    )
    metadata["accelerator"] = "GPU"
    notebook_cells = []
    for index, cell in enumerate(cells):
        payload = {
            "cell_type": cell["cell_type"],
            "id": f"plan-c-{index:02d}",
            "metadata": {},
            "source": source_lines(cell["source"]),
        }
        if cell["cell_type"] == "code":
            payload.update({"execution_count": None, "outputs": []})
        notebook_cells.append(payload)
    TARGET.write_text(
        json.dumps(
            {
                "cells": notebook_cells,
                "metadata": metadata,
                "nbformat": 4,
                "nbformat_minor": 5,
            },
            ensure_ascii=False,
            indent=1,
        )
        + "\n",
        encoding="utf-8",
    )


def verify_target() -> None:
    notebook = json.loads(TARGET.read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    assert notebook["nbformat_minor"] == 5
    assert len(notebook["cells"]) == 21
    all_source = "\n".join(
        "".join(cell.get("source", [])) for cell in notebook["cells"]
    )
    assert "�" not in all_source
    assert "wandb_v1_" not in all_source
    for index, cell in enumerate(notebook["cells"]):
        if cell["cell_type"] == "code":
            assert cell["execution_count"] is None
            assert cell["outputs"] == []
            compile("".join(cell["source"]), f"target-cell-{index}", "exec")

    training_cell = "".join(notebook["cells"][16]["source"])
    assert training_cell.strip().endswith(
        "EXPERIMENT_RESULTS = run_required_experiments()"
    )
    assert len(training_cell.splitlines()) == 2

    required = [
        "reporting_seeds: tuple[int, ...] = (3407, 3408, 3409)",
        'systems: tuple[str, ...] = ("A2", "A3", "A4", "A5", "C")',
        '"no_global_encoder"',
        '"no_boundary_filter"',
        '"no_semicrf"',
        '"no_coarse_curriculum"',
        '"no_presence_loss"',
        '"no_monotonic_mask"',
        '"no_stage3"',
        "def unitize_text(",
        "def prepare_inference_document(",
        "unlabeled_units = unitize_text(",
        "def attach_gold_supervision(",
        "label_truncations",
        "gold_roundtrip_ok",
        "def stage1_document_backward(",
        "class WeightedLossMean:",
        '"train/loss": shared_loss',
        'wandb.define_metric("global_step")',
        'wandb.define_metric(f"{metric_namespace}/*", step_metric="global_step")',
        "wandb.watch(",
        "def restore_best_weights(",
        "restore_best_weights()",
        "def restore_embedding_cache_artifact(",
        "restore_embedding_cache_artifact()",
        "def load_or_prepare_split(",
        "restore_prepared_documents_artifact()",
        "log_prepared_documents_artifact()",
        "def required_experiment_specs(",
        "def measured_promotion_gates(",
        'AUDIT[split]["label_truncations"] == 0',
        'AUDIT[split]["roundtrip_rows"] == AUDIT[split]["raw_rows"]',
    ]
    missing = [text for text in required if text not in all_source]
    assert not missing, f"Missing required implementation markers: {missing}"

    unitizer = all_source.split("def unitize_text(", 1)[1].split(
        "def attach_gold_supervision(", 1
    )[0]
    assert "GoldSpan" not in unitizer
    assert "sections_json" not in unitizer
    assert '"no_label_truncation": True' not in all_source
    assert '"gold_roundtrip_100pct": True' not in all_source
    assert 'os.environ["WANDB_WATCH"] = "false"' not in all_source
    print("Target notebook verification passed.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--verify-target", action="store_true")
    args = parser.parse_args()
    cells = parse_percent_script(SOURCE.read_text(encoding="utf-8"))
    summarize(cells)
    if args.write or args.check:
        clean_cells = clean_layout(cells)
        print("\nClean notebook layout:")
        summarize(clean_cells)
    if args.check:
        for index, cell in enumerate(clean_cells):
            if cell["cell_type"] == "code":
                compile(cell["source"], f"notebook-cell-{index}", "exec")
        print("All code cells compile.")
    if args.write:
        write_notebook(clean_cells)
    if args.verify_target:
        verify_target()


if __name__ == "__main__":
    main()
