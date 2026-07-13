"""
Add four hand-curated reference papers that the broad search missed
(or that the rate-limited Semantic Scholar queries failed to surface).

The four:
  1. DRAGON  - arXiv:2602.09071 (GitHub-repo classification, 825K corpus)
  2. Snorkel - Ratner et al. 2017 VLDB (weak-supervision foundational paper)
  3. Stronger Than You Think - NeurIPS 2024 D&B (LLM-LF weak-supervision benchmark)
  4. Gilardi et al. 2023 - arXiv:2303.15056 (ChatGPT vs crowd-workers, PNAS)

Reads:  paper/literature/candidates.json  (the existing search results)
Writes: paper/literature/candidates.json  (appends "known_additions" block)
        paper/literature/pdfs/<id>.pdf    (where downloadable)
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import requests


LIT_DIR = Path(__file__).resolve().parent
PDF_DIR = LIT_DIR / "pdfs"
CANDIDATES_JSON = LIT_DIR / "candidates.json"

USER_AGENT = (
    "NAICS-GH-paper-litreview/0.1 "
    "(https://huggingface.co/datasets/alexanderquispe/naics-github-classifier; "
    "alexanderquispe@github.com)"
)


# -------------------------------------------------------------------- helpers


def fetch_arxiv_metadata(arxiv_id: str) -> dict:
    """Query the arXiv API for metadata on a single ID."""
    import xml.etree.ElementTree as ET

    url = f"https://export.arxiv.org/api/query?id_list={arxiv_id}"
    ns = {"atom": "http://www.w3.org/2005/Atom"}

    delay = 6.0
    for attempt in range(1, 6):
        try:
            r = requests.get(
                url, timeout=60, headers={"User-Agent": USER_AGENT}
            )
            if r.status_code in (429, 503):
                print(f"  [arxiv {arxiv_id}] HTTP {r.status_code}, sleeping {delay:.1f}s")
                time.sleep(delay)
                delay = min(delay * 2, 60.0)
                continue
            r.raise_for_status()
            root = ET.fromstring(r.content)
            entry = root.find("atom:entry", ns)
            if entry is None:
                return {}
            title = (
                entry.findtext("atom:title", default="", namespaces=ns) or ""
            ).strip().replace("\n", " ")
            title = re.sub(r"\s+", " ", title)
            abstract = (
                entry.findtext("atom:summary", default="", namespaces=ns) or ""
            ).strip().replace("\n", " ")
            abstract = re.sub(r"\s+", " ", abstract)
            published = entry.findtext("atom:published", default="", namespaces=ns) or ""
            year = None
            m = re.match(r"(\d{4})", published)
            if m:
                year = int(m.group(1))
            authors = [
                (a.findtext("atom:name", default="", namespaces=ns) or "").strip()
                for a in entry.findall("atom:author", ns)
            ]
            return {
                "title": title,
                "authors": authors,
                "year": year,
                "abstract": abstract[:600],
            }
        except Exception as e:
            print(f"  [arxiv {arxiv_id}] attempt {attempt} failed: {e}")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    return {}


def download_pdf(url: str, out_name: str,
                  max_bytes: int = 30 * 1024 * 1024) -> Path | None:
    """Best-effort PDF download to PDF_DIR / out_name."""
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PDF_DIR / out_name
    if out_path.exists() and out_path.stat().st_size > 4096:
        print(f"  [pdf] already on disk: {out_name}")
        return out_path
    try:
        r = requests.get(
            url, stream=True, timeout=60,
            headers={"User-Agent": USER_AGENT},
        )
        r.raise_for_status()
        ctype = (r.headers.get("Content-Type") or "").lower()
        if "pdf" not in ctype and "octet-stream" not in ctype:
            print(f"  [pdf] skip non-pdf content-type for {url} ({ctype})")
            return None
        total = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=64 * 1024):
                if not chunk:
                    continue
                total += len(chunk)
                if total > max_bytes:
                    print(f"  [pdf] too large, aborting: {url}")
                    f.close()
                    out_path.unlink(missing_ok=True)
                    return None
                f.write(chunk)
        print(f"  [pdf] saved {out_name} ({total/1024:.0f} KB)")
        return out_path
    except Exception as e:
        print(f"  [pdf] failed {url}: {e}")
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        return None


# -------------------------------------------------------------------- entries


def build_dragon() -> dict:
    """DRAGON: arXiv:2602.09071."""
    meta = fetch_arxiv_metadata("2602.09071")
    return {
        "source": "manual_known_addition",
        "title": meta.get("title")
            or "DRAGON: Robust Classification for Very Large Collections of Software Repositories",
        "authors": meta.get("authors") or [],
        "year": meta.get("year") or 2026,
        "venue": "arXiv",
        "citations": None,
        "abstract": (
            meta.get("abstract")
            or "DRAGON releases the largest open dataset to date for repository "
            "classification: 825K repositories with ground-truth topics from the "
            "Software Heritage archive. F1@5 = 60.8%, robust to missing READMEs."
        ),
        "arxiv_id": "2602.09071",
        "doi": None,
        "openaccess_pdf": "https://arxiv.org/pdf/2602.09071.pdf",
        "_why_added": "Largest open GitHub-repo-classification dataset (825K). "
            "Critical Related-Work anchor for §2.1.",
    }


def build_snorkel() -> dict:
    """Snorkel (Ratner et al. 2017, VLDB)."""
    return {
        "source": "manual_known_addition",
        "title": "Snorkel: Rapid Training Data Creation with Weak Supervision",
        "authors": [
            "Alexander Ratner", "Stephen H. Bach", "Henry Ehrenberg",
            "Jason Fries", "Sen Wu", "Christopher Ré",
        ],
        "year": 2017,
        "venue": "Proceedings of the VLDB Endowment",
        "citations": None,
        "abstract": (
            "Labeling training data is increasingly the largest bottleneck in "
            "deploying machine learning systems. We present Snorkel, a "
            "first-of-its-kind system that enables users to train state-of-the-art "
            "models without hand labeling any training data. Instead, users write "
            "labeling functions that express arbitrary heuristics, which can have "
            "unknown accuracies and correlations. Snorkel denoises their outputs "
            "without access to ground truth by incorporating the first end-to-end "
            "implementation of our recently proposed machine learning paradigm, "
            "data programming."
        ),
        "arxiv_id": "1711.10160",
        "doi": "10.14778/3157794.3157797",
        "openaccess_pdf": "https://arxiv.org/pdf/1711.10160.pdf",
        "_why_added": "Foundational weak-supervision / labeling-functions paper. "
            "Anchor citation for §2.2 (LLM-as-labeler is a special case).",
    }


def build_stronger() -> dict:
    """Stronger Than You Think (NeurIPS 2024 D&B)."""
    return {
        "source": "manual_known_addition",
        "title": "Stronger Than You Think: Benchmarking Weak Supervision on Realistic Tasks",
        "authors": [],   # filled in from PDF if needed
        "year": 2024,
        "venue": "NeurIPS 2024 Datasets and Benchmarks Track",
        "citations": None,
        "abstract": (
            "A benchmark of weak-supervision pipelines on realistic tasks, "
            "comparing handwritten labeling functions, programmatic labeling, "
            "and LLM-generated labeling functions. Shows that LLM-LFs can match "
            "or exceed handcrafted LFs on several realistic benchmarks; informs "
            "the LLM-as-labeler design space."
        ),
        "arxiv_id": None,
        "doi": None,
        "openaccess_pdf": (
            "https://proceedings.neurips.cc/paper_files/paper/2024/file/"
            "dd26d03d50af993ed052578c730e9729-Paper-Datasets_and_Benchmarks_Track.pdf"
        ),
        "_why_added": "Closest NeurIPS D&B 2024 paper to ours methodologically. "
            "Important precedent for §2.2 and §2.4.",
    }


def build_gilardi() -> dict:
    """Gilardi, Alizadeh, Kubli 2023 — arXiv:2303.15056 + PNAS."""
    meta = fetch_arxiv_metadata("2303.15056")
    return {
        "source": "manual_known_addition",
        "title": meta.get("title")
            or "ChatGPT outperforms crowd-workers for text-annotation tasks",
        "authors": meta.get("authors") or [
            "Fabrizio Gilardi", "Meysam Alizadeh", "Maël Kubli",
        ],
        "year": meta.get("year") or 2023,
        "venue": "Proceedings of the National Academy of Sciences (PNAS)",
        "citations": None,
        "abstract": (
            meta.get("abstract")
            or "Many NLP applications require manual data annotations for a variety "
            "of tasks. To facilitate such annotation, recent research has compared "
            "MTurk crowd-workers with ChatGPT and found ChatGPT exhibits "
            "remarkable performance, both in terms of accuracy and agreement, "
            "while costing a fraction of the price."
        ),
        "arxiv_id": "2303.15056",
        "doi": "10.1073/pnas.2305016120",
        "openaccess_pdf": "https://arxiv.org/pdf/2303.15056.pdf",
        "_why_added": "Canonical citation for 'LLMs as annotators outperform "
            "crowd-workers' — anchors the methodological choice in §3.4.",
    }


# -------------------------------------------------------------------- main


def main() -> None:
    if not CANDIDATES_JSON.exists():
        raise SystemExit(
            f"{CANDIDATES_JSON} not found — run build_related_work.py first."
        )

    candidates = json.loads(CANDIDATES_JSON.read_text(encoding="utf-8"))

    print("Building known-addition entries...")
    entries = []
    for builder in (build_dragon, build_snorkel, build_stronger, build_gilardi):
        e = builder()
        entries.append(e)
        # Try to download the PDF
        if e.get("openaccess_pdf") and e.get("arxiv_id"):
            arxiv_id = re.sub(r"v\d+$", "", e["arxiv_id"])
            path = download_pdf(e["openaccess_pdf"], f"{arxiv_id}.pdf")
            if path is not None:
                e["local_pdf"] = str(path.relative_to(LIT_DIR))
        elif e.get("openaccess_pdf"):
            # NeurIPS proceedings PDF — name by leading hash from URL
            url = e["openaccess_pdf"]
            slug = re.sub(r"[^a-z0-9]+", "_",
                          (e.get("title") or "paper").lower())[:60]
            path = download_pdf(url, f"{slug}.pdf")
            if path is not None:
                e["local_pdf"] = str(path.relative_to(LIT_DIR))
        time.sleep(4.0)   # be polite

    # Append as a new block
    new_block = {
        "block": "known_additions",
        "label": "Manually added references not surfaced by the API search",
        "ss_query": None,
        "arxiv_query": None,
        "papers": entries,
    }

    # Drop any pre-existing block of the same name and re-append
    candidates = [b for b in candidates if b.get("block") != "known_additions"]
    candidates.append(new_block)

    CANDIDATES_JSON.write_text(
        json.dumps(candidates, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nUpdated {CANDIDATES_JSON}")
    n_with_pdf = sum(1 for e in entries if e.get("local_pdf"))
    print(
        f"  {len(entries)} known additions appended, "
        f"{n_with_pdf} with local PDF."
    )


if __name__ == "__main__":
    main()
