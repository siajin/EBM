"""Retrieve public research papers for a full-text review; never loads lab data."""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import json
import requests
import fitz

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "artifacts/literature/ebm_channels_20261105"
OUT.mkdir(parents=True, exist_ok=True)

PAPERS = [
    ("01_ebm_2025", "https://www.isca-archive.org/interspeech_2025/kim25g_interspeech.pdf"),
    ("02_ebm_journal_2026", "https://www.researchgate.net/publication/403465905_Comprehensive_Validation_of_Bridge_Module_and_EBM_Loss_for_One-Class_Audio_Deepfake_Detection/fulltext/6a55ae497fcc0765f3df7f76/Comprehensive-Validation-of-Bridge-Module-and-EBM-Loss-for-One-Class-Audio-Deepfake-Detection.pdf"),
    ("03_ocsoftmax_2021", "https://arxiv.org/pdf/2010.13995"),
    ("04_acs_2024", "https://www.isca-archive.org/interspeech_2024/kim24b_interspeech.pdf"),
    ("05_fisher_2011", "https://web.cs.ucla.edu/~qgu/pdf/uai11.pdf"),
    ("06_dice_2022", "https://arxiv.org/pdf/2111.09805"),
    ("07_ash_2023", "https://openreview.net/pdf?id=ndYXTEL6cZz"),
    ("08_rankfeat_2022", "https://arxiv.org/pdf/2209.08590"),
    ("09_matryoshka_2022", "https://arxiv.org/pdf/2205.13147"),
    ("10_neural_collapse_2020", "https://arxiv.org/pdf/2008.08186"),
    ("11_cosine_2024", "https://arxiv.org/pdf/2403.05440"),
    ("12_silence_2021", "https://arxiv.org/pdf/2106.12914"),
    ("13_shortcuts_2025", "https://openaccess.thecvf.com/content/CVPR2025/papers/Smeu_Circumventing_Shortcuts_in_Audio-visual_Deepfake_Detection_Datasets_with_Unsupervised_Learning_CVPR_2025_paper.pdf"),
    ("14_three_features_2026", "https://cdn.techscience.cn/files/cmc/2026/TSP_CMC-88-3/TSP_CMC_80593/TSP_CMC_80593.pdf"),
    ("15_comprehensive_ood_2024", "https://arxiv.org/pdf/2401.10176"),
]


def get_paper(pair):
    key, url = pair
    pdf, txt = OUT / (key + ".pdf"), OUT / (key + ".txt")
    try:
        if not pdf.exists():
            r = requests.get(url, timeout=45, headers={"User-Agent":"Mozilla/5.0 (research literature review)"})
            r.raise_for_status()
            if not r.content.startswith(b"%PDF"):
                raise ValueError("Returned content is not a PDF")
            pdf.write_bytes(r.content)
        document = fitz.open(pdf)
        pages = [p.get_text(sort=True) for p in document]
        txt.write_text("\n".join(f"\n===== PDF PAGE {i+1} =====\n{text}" for i, text in enumerate(pages)), encoding="utf-8")
        return dict(key=key, url=url, status="full_text_available", pages=len(pages),
                    text_chars=sum(map(len,pages)), pdf_sha256=hashlib.sha256(pdf.read_bytes()).hexdigest(),
                    extracted_text_sha256=hashlib.sha256(txt.read_bytes()).hexdigest(),
                    local_pdf=str(pdf.relative_to(ROOT)), local_text=str(txt.relative_to(ROOT)))
    except Exception as e:
        return dict(key=key, url=url, status="not_retrieved", error=str(e))


def main():
    records = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        for job in as_completed([pool.submit(get_paper, pair) for pair in PAPERS]):
            record = job.result()
            records.append(record)
            print(json.dumps(record, ensure_ascii=False), flush=True)
    records.sort(key=lambda r:r["key"])
    (OUT / "retrieval_manifest.json").write_text(json.dumps(dict(
        retrieved_at_utc=datetime.now(timezone.utc).isoformat(), papers=records,
        note="Retrieval is not evidence of reading; section-level reading notes tracked separately."),
        ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
