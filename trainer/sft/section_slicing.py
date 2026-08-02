"""Gold-span section slicing shared by the memory-efficient SFT pipeline.

This mirrors the evaluation-unit construction in
``notebook/precompute_qwen3_5_section_eval.py``: one parent decision becomes
one example per canonical section, and the user input contains only that
section's annotated source spans.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence
import json


SECTION_GUIDANCE: dict[str, str] = {
    "judul": "Ambil judul PUTUSAN/P U T U S A N di awal dokumen; berhenti sebelum nomor perkara.",
    "nomor_putusan": "Ambil nomor perkara yang mengikuti judul; jangan sertakan nama pengadilan atau irah-irah.",
    "irah_irah": "Ambil formula DEMI KEADILAN BERDASARKAN KETUHANAN YANG MAHA ESA, termasuk variasi spasi/OCR.",
    "nama_pengadilan_negeri": "Ambil nama pengadilan tingkat pertama pada kalimat pembuka.",
    "keterangan_perkara": "Ambil uraian bahwa pengadilan mengadili perkara sampai sebelum identitas.",
    "nama_lengkap": "Ambil hanya nilai identitas pada label Nama lengkap dan pertahankan semua subjek berurutan.",
    "tempat_lahir": "Ambil hanya nilai Tempat lahir untuk setiap subjek.",
    "umur_tanggal_lahir": "Ambil hanya nilai umur dan/atau tanggal lahir untuk setiap subjek.",
    "jenis_kelamin": "Ambil hanya nilai Jenis kelamin untuk setiap subjek.",
    "kebangsaan": "Ambil hanya nilai Kebangsaan/Kewarganegaraan/nasionalitas.",
    "tempat_tinggal": "Ambil hanya alamat Tempat tinggal untuk setiap subjek.",
    "agama": "Ambil hanya nilai Agama untuk setiap subjek.",
    "pekerjaan": "Ambil hanya nilai Pekerjaan untuk setiap subjek.",
    "penangkapan": "Ambil hanya penangkapan beserta tanggal dan referensinya; jangan masukkan penahanan.",
    "penahanan": "Ambil seluruh tahap/perpanjangan penahanan, termasuk penangguhan, pembantaran, dan pengalihan.",
    "tuntutan": "Ambil lengkap seluruh amar tuntutan bernomor sampai sebelum pembelaan atau dakwaan berikutnya.",
    "dakwaan": "Ambil lengkap semua bentuk dakwaan beserta uraian perbuatan dan pasal.",
    "saksi": "Ambil seluruh keterangan saksi, termasuk nama, sumpah, butir keterangan, dan tanggapan subjek.",
    "ahli": "Ambil seluruh keterangan ahli penuntut maupun pembela; jangan isi dari saksi biasa.",
    "terdakwa": "Ambil keterangan Terdakwa/Para Terdakwa atau Anak sendiri di persidangan.",
    "surat": "Ambil alat bukti surat, dokumen, dan elektronik; jangan campur daftar barang bukti fisik.",
    "petunjuk_barang_bukti": "Ambil inventaris barang bukti yang diajukan Penuntut Umum.",
    "fakta_hukum": "Ambil daftar fakta hukum dan berhenti sebelum analisis hukum atau unsur.",
    "pertimbangan_hukum": "Ambil seluruh analisis hukum Majelis sampai sebelum MENGADILI.",
    "amar_putusan": "Ambil setiap perintah mulai MENGADILI sampai sebelum Demikianlah diputuskan.",
    "hari": "Ambil nama hari musyawarah putusan, bukan hari pengucapan bila berbeda.",
    "tanggal": "Ambil tanggal musyawarah putusan persis seperti span sumber.",
    "tahun": "Ambil hanya tahun musyawarah putusan dari formula penutup.",
    "siapa_yang_memutus": "Salin seluruh span tentang hakim yang memutus.",
    "panitera_pengganti": "Ambil nama Panitera/Panitera Pengganti yang membantu persidangan.",
    "tanda_tangan_majelis": "Ambil blok tanda tangan/nama Majelis dan Panitera pada akhir dokumen.",
}

COMMON_CONTRACT_TEMPLATE = """Anda mengekstrak SATU bagian putusan pengadilan Indonesia dari teks sumber yang sudah dipotong khusus untuk bagian tersebut.
Bagian yang diminta: {section}.
Keluarkan SATU objek JSON saja, tanpa markdown, penjelasan, analisis, reasoning, atau teks lain.
Bentuk wajib: {{"sections": {{"{section}": ["kutipan"]}}, "empty_sections": []}}.
Objek sections harus berisi tepat satu kunci, yaitu {section}; jangan keluarkan 30 kunci lain.
Teks sumber berisi nol atau lebih blok <span>...</span>. Salin seluruh isi setiap blok sebagai tepat satu item array, pertahankan urutannya, dan jangan sertakan tag <span>.
Setiap nilai harus array string. Setiap string harus kutipan verbatim dan kontigu dari teks sumber: jangan meringkas, memparafrasekan, memperbaiki OCR, menormalkan ejaan/spasi, menggabungkan potongan tak-kontigu, atau mengarang.
Jika ada beberapa span terpisah, pertahankan sebagai beberapa item array dalam urutan sumber.
Jika teks sumber kosong atau bagian tidak ada, gunakan [] dan isi empty_sections dengan ["{section}"]. Jika ada kutipan, empty_sections harus [].

Panduan bagian ini:
{guidance}
"""

CORPUS_ADDENDA = {
    "Anak": "Korpus Anak: subjek adalah Anak, bukan Terdakwa dewasa. Pertahankan identitas jamak dan seluruh tahap LPAS/LPKS berurutan.",
    "Asusila": "Korpus Asusila/Pidana Biasa: subjek adalah Terdakwa/Para Terdakwa. Pertahankan semua identitas, bentuk dakwaan, dan tahap penahanan berurutan.",
    "TPPO": "Korpus TPPO: pertahankan semua Terdakwa, tahap penahanan, restitusi, disposisi barang bukti, dan perintah amar berurutan.",
}


def _target_sections(value: Any) -> Mapping[str, Sequence[str]]:
    target = json.loads(value) if isinstance(value, str) else value
    if not isinstance(target, Mapping) or not isinstance(
        target.get("sections"), Mapping
    ):
        raise ValueError("target_json must contain a sections object")
    sections = target["sections"]
    if set(sections) != set(SECTION_GUIDANCE):
        raise ValueError("target_json does not contain the 31 canonical sections")
    return sections


def slice_row_by_section(
    row: Mapping[str, Any], source_row_no: int | None = None
) -> list[dict[str, Any]]:
    """Expand one whole-document row into the notebook's 31 section units."""

    sections = _target_sections(row.get("target_json"))
    corpus = str(row.get("corpus", ""))
    if corpus not in CORPUS_ADDENDA:
        raise ValueError(f"Unknown corpus {corpus!r}")
    parent_id = str(row.get("id", row.get("parent_id", "")))
    children: list[dict[str, Any]] = []
    for section, guidance in SECTION_GUIDANCE.items():
        raw_spans = sections[section]
        if not isinstance(raw_spans, Sequence) or isinstance(raw_spans, str):
            raise ValueError(f"Section {section!r} must be an array of strings")
        spans = [str(span) for span in raw_spans]
        sliced_input = "\n".join(
            f"<span>\n{span}\n</span>" for span in spans
        )
        source = sliced_input or "[TIDAK ADA BLOK <span>]"
        question = (
            f"Bagian yang diminta: {section}.\n\nTEKS SUMBER:\n{source}"
        )
        system_prompt = (
            COMMON_CONTRACT_TEMPLATE.format(
                section=section, guidance=guidance
            )
            + "\n"
            + CORPUS_ADDENDA[corpus]
        )
        answer = json.dumps(
            {
                "sections": {section: spans},
                "empty_sections": [section] if not spans else [],
            },
            ensure_ascii=False,
        )
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]
        dataset_id = (
            f"row-{source_row_no:06d}::{parent_id}::{section}"
            if source_row_no is not None
            else f"{parent_id}::{section}"
        )
        children.append(
            {
                "id": dataset_id,
                "parent_id": parent_id,
                "corpus": corpus,
                "section": section,
                "span_count": len(spans),
                "is_empty": not spans,
                "sliced_input": sliced_input,
                "target_json": answer,
                "messages": messages,
                "prompt": messages[:2],
                "answer": answer,
            }
        )
    return children


def slice_batch_by_section(
    examples: Mapping[str, Sequence[Any]], indices: Sequence[int] | None = None
) -> dict[str, list[Any]]:
    """Batched Hugging Face ``Dataset.map`` adapter for section slicing."""

    keys = list(examples)
    if not keys:
        return {}
    row_count = len(examples[keys[0]])
    children: list[dict[str, Any]] = []
    for index in range(row_count):
        source_row_no = int(indices[index]) + 1 if indices is not None else None
        children.extend(slice_row_by_section(
            {key: examples[key][index] for key in keys}, source_row_no
        ))
    output_keys = list(children[0]) if children else []
    return {key: [child[key] for child in children] for key in output_keys}
