"""
Literature review builder for NAICS-GH paper.

Inspired by Top_papers_creator/idea-evaluation-pipeline (Stage 3 step 5).
Differences from that pipeline:
  - Queries both Semantic Scholar AND arXiv (their pipeline uses only SS).
  - Downloads PDFs when an arXiv ID is available.
  - Tailored queries for a dataset/benchmark paper (not causal econ).

Usage:
    python paper/literature/build_related_work.py
        --> writes  paper/literature/candidates.json
        --> writes  paper/literature/pdfs/{arxiv_id}.pdf
        --> writes  paper/literature/summary.md
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path
from urllib.parse import quote_plus

import requests


# ----- Query definitions -----------------------------------------------------

# Each query maps to one block of our Related Work section.
QUERIES = [
    {
        "block": "github_classification",
        "label": "GitHub repository classification (closest prior work)",
        "ss_query": "GitHub repository classification topic application domain",
        "arxiv_query": "GitHub repository classification topic application domain",
    },
    {
        "block": "weak_supervision_llm",
        "label": "LLM-as-labeler / weak supervision for dataset construction",
        "ss_query": "LLM weak supervision dataset annotation GPT-4 labeler",
        "arxiv_query": "weak supervision LLM annotation labeling dataset construction",
    },
    {
        "block": "industry_classification_text",
        "label": "Industry classification (NAICS) applied to text or non-standard data",
        "ss_query": "NAICS industry classification text machine learning",
        "arxiv_query": "industry classification NAICS company text deep learning",
    },
    {
        "block": "retrieve_and_verify",
        "label": "Retrieve-and-verify pipelines for labeled dataset generation",
        "ss_query": "retrieve verify LLM dataset construction labeling pipeline",
        "arxiv_query": "retrieve verify LLM weak labeling embedding retrieval dataset",
    },
]

PER_QUERY_LIMIT = 8         # Top-k per source per query
PDF_DOWNLOAD_LIMIT = 30      # Cap on number of PDFs we'll download
USER_AGENT = (
    "NAICS-GH-paper-litreview/0.1 "
    "(https://huggingface.co/datasets/alexanderquispe/naics-github-classifier; "
    "alexanderquispe@github.com)"
)

# ----- Paths -----------------------------------------------------------------

LIT_DIR = Path(__file__).resolve().parent
PDF_DIR = LIT_DIR / "pdfs"
CANDIDATES_JSON = LIT_DIR / "candidates.json"
SUMMARY_MD = LIT_DIR / "summary.md"


# ----- Semantic Scholar ------------------------------------------------------

def _request_with_retry(url: str, params: dict | None = None, *,
                         label: str, timeout: int = 60,
                         max_attempts: int = 5,
                         initial_backoff: float = 5.0) -> requests.Response | None:
    """GET with exponential backoff on 429/5xx and connection errors."""
    delay = initial_backoff
    for attempt in range(1, max_attempts + 1):
        try:
            r = requests.get(
                url, params=params, timeout=timeout,
                headers={"User-Agent": USER_AGENT},
            )
            if r.status_code == 429 or 500 <= r.status_code < 600:
                wait = float(r.headers.get("Retry-After") or delay)
                print(f"  [{label}] HTTP {r.status_code}, attempt {attempt}/{max_attempts}; "
                      f"sleeping {wait:.1f}s")
                time.sleep(wait)
                delay = min(delay * 2, 60.0)
                continue
            r.raise_for_status()
            return r
        except requests.exceptions.RequestException as e:
            print(f"  [{label}] attempt {attempt}/{max_attempts} failed: {e}; "
                  f"sleeping {delay:.1f}s")
            time.sleep(delay)
            delay = min(delay * 2, 60.0)
    print(f"  [{label}] giving up on {url} after {max_attempts} attempts.")
    return None


def search_semantic_scholar(query: str, limit: int = PER_QUERY_LIMIT) -> list[dict]:
    """Query Semantic Scholar Graph API for papers.

    Returns: title, authors, year, venue, citationCount, abstract, externalIds.
    No API key required.
    """
    url = "https://api.semanticscholar.org/graph/v1/paper/search"
    params = {
        "query": query,
        "limit": limit,
        "fields": "title,authors,year,venue,citationCount,abstract,externalIds,openAccessPdf",
    }
    r = _request_with_retry(url, params, label="semantic-scholar")
    if r is None:
        return []
    try:
        data = r.json().get("data", []) or []
    except Exception as e:
        print(f"  [semantic-scholar] JSON parse failed: {e}")
        return []
    out = []
    for p in data:
        authors = [a.get("name", "") for a in (p.get("authors") or [])]
        ext = p.get("externalIds") or {}
        out.append({
            "source": "semantic_scholar",
            "title": p.get("title") or "",
            "authors": authors,
            "year": p.get("year"),
            "venue": p.get("venue") or "",
            "citations": p.get("citationCount") or 0,
            "abstract": (p.get("abstract") or "")[:600],
            "arxiv_id": ext.get("ArXiv"),
            "doi": ext.get("DOI"),
            "openaccess_pdf": (p.get("openAccessPdf") or {}).get("url"),
        })
    return out


# ----- arXiv -----------------------------------------------------------------

ARXIV_NS = {"atom": "http://www.w3.org/2005/Atom"}


def search_arxiv(query: str, limit: int = PER_QUERY_LIMIT) -> list[dict]:
    """Query the arXiv Atom API.

    Returns: title, authors, year, abstract, arxiv_id, pdf_url.
    """
    import xml.etree.ElementTree as ET

    url = (
        "https://export.arxiv.org/api/query"
        f"?search_query=all:{quote_plus(query)}"
        f"&start=0&max_results={limit}"
        "&sortBy=relevance&sortOrder=descending"
    )
    r = _request_with_retry(url, label="arxiv", timeout=60, initial_backoff=6.0)
    if r is None:
        return []
    try:
        root = ET.fromstring(r.content)
        out = []
        for entry in root.findall("atom:entry", ARXIV_NS):
            title = (entry.findtext("atom:title", default="", namespaces=ARXIV_NS) or "").strip().replace("\n", " ")
            title = re.sub(r"\s+", " ", title)
            abstract = (entry.findtext("atom:summary", default="", namespaces=ARXIV_NS) or "").strip().replace("\n", " ")
            abstract = re.sub(r"\s+", " ", abstract)
            published = entry.findtext("atom:published", default="", namespaces=ARXIV_NS) or ""
            year = None
            m = re.match(r"(\d{4})", published)
            if m:
                year = int(m.group(1))
            authors = [
                (a.findtext("atom:name", default="", namespaces=ARXIV_NS) or "").strip()
                for a in entry.findall("atom:author", ARXIV_NS)
            ]
            id_url = entry.findtext("atom:id", default="", namespaces=ARXIV_NS) or ""
            m2 = re.search(r"abs/([^/v]+)(?:v\d+)?$", id_url)
            arxiv_id = m2.group(1) if m2 else None
            pdf_url = None
            for link in entry.findall("atom:link", ARXIV_NS):
                if link.get("title") == "pdf":
                    pdf_url = link.get("href")
                    break
            if pdf_url is None and arxiv_id:
                pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
            out.append({
                "source": "arxiv",
                "title": title,
                "authors": authors,
                "year": year,
                "venue": "arXiv",
                "citations": None,
                "abstract": abstract[:600],
                "arxiv_id": arxiv_id,
                "doi": None,
                "openaccess_pdf": pdf_url,
            })
        return out
    except Exception as e:
        print(f"  [arxiv] '{query}' -> XML parse ERROR: {e}")
        return []


# ----- Dedup -----------------------------------------------------------------

def normalize_title(title: str) -> str:
    return re.sub(r"[^\w\s]", "", (title or "").lower()).strip()


def dedupe(records: list[dict]) -> list[dict]:
    """Dedupe on (normalized title, year). Prefer the record that has more
    information (arxiv_id, then DOI, then citation count)."""
    by_key: dict[tuple[str, int | None], dict] = {}
    for rec in records:
        key = (normalize_title(rec.get("title") or ""), rec.get("year"))
        if not key[0]:
            continue
        cur = by_key.get(key)
        if cur is None:
            by_key[key] = rec
            continue
        # Merge: prefer entry with more identifiers
        def score(r):
            return (
                int(bool(r.get("arxiv_id"))),
                int(bool(r.get("doi"))),
                r.get("citations") or 0,
            )
        if score(rec) > score(cur):
            # Carry over abstract and citations if missing in the better record
            if not rec.get("abstract") and cur.get("abstract"):
                rec["abstract"] = cur["abstract"]
            if rec.get("citations") in (None, 0) and cur.get("citations"):
                rec["citations"] = cur["citations"]
            by_key[key] = rec
    return list(by_key.values())


# ----- PDF download ----------------------------------------------------------

def download_pdf(record: dict, out_dir: Path, max_bytes: int = 30 * 1024 * 1024) -> Path | None:
    """Best-effort PDF download. Returns the saved path or None."""
    arxiv_id = record.get("arxiv_id")
    url = None
    fname = None

    if arxiv_id:
        # Normalize arxiv_id: strip version
        clean_id = re.sub(r"v\d+$", "", arxiv_id)
        url = f"https://arxiv.org/pdf/{clean_id}.pdf"
        fname = f"{clean_id}.pdf"
    elif record.get("openaccess_pdf"):
        url = record["openaccess_pdf"]
        # Build filename from title hash-ish
        slug = re.sub(r"[^a-z0-9]+", "_", (record.get("title") or "paper").lower())[:60]
        fname = f"{slug}.pdf"

    if not url or not fname:
        return None

    out_path = out_dir / fname
    if out_path.exists() and out_path.stat().st_size > 4096:
        return out_path

    try:
        r = requests.get(url, stream=True, timeout=60, headers={"User-Agent": USER_AGENT})
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
        print(f"  [pdf] saved {out_path.name} ({total/1024:.0f} KB)")
        return out_path
    except Exception as e:
        print(f"  [pdf] failed {url}: {e}")
        if out_path.exists():
            try:
                out_path.unlink()
            except Exception:
                pass
        return None


# ----- Main ------------------------------------------------------------------

def main() -> None:
    LIT_DIR.mkdir(parents=True, exist_ok=True)
    PDF_DIR.mkdir(parents=True, exist_ok=True)

    all_blocks = []
    seen_pdf_targets: set[str] = set()
    pdf_count = 0

    for q in QUERIES:
        print(f"\n=== {q['label']} ===")

        ss = search_semantic_scholar(q["ss_query"], limit=PER_QUERY_LIMIT)
        print(f"  semantic_scholar -> {len(ss)} hits for '{q['ss_query']}'")
        time.sleep(5.0)  # SS public endpoint throttles aggressively without an API key

        ar = search_arxiv(q["arxiv_query"], limit=PER_QUERY_LIMIT)
        print(f"  arxiv -> {len(ar)} hits for '{q['arxiv_query']}'")
        time.sleep(5.0)  # arXiv asks for at least 3s between requests; give margin

        merged = dedupe(ss + ar)
        # Sort: prefer cited + recent
        merged.sort(
            key=lambda r: ((r.get("citations") or 0), (r.get("year") or 0)),
            reverse=True,
        )

        # Download PDFs (capped)
        for rec in merged:
            if pdf_count >= PDF_DOWNLOAD_LIMIT:
                break
            key = rec.get("arxiv_id") or rec.get("openaccess_pdf")
            if not key or key in seen_pdf_targets:
                continue
            seen_pdf_targets.add(key)
            path = download_pdf(rec, PDF_DIR)
            if path is not None:
                rec["local_pdf"] = str(path.relative_to(LIT_DIR))
                pdf_count += 1

        all_blocks.append({
            "block": q["block"],
            "label": q["label"],
            "ss_query": q["ss_query"],
            "arxiv_query": q["arxiv_query"],
            "papers": merged,
        })

    # Write the candidates JSON
    CANDIDATES_JSON.write_text(
        json.dumps(all_blocks, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {CANDIDATES_JSON}")

    # Write a human-readable summary
    lines = ["# Literature search results", ""]
    lines.append(
        "Auto-generated by `paper/literature/build_related_work.py`. "
        "Edit nothing here by hand — re-run the script to refresh.\n"
    )
    for block in all_blocks:
        lines.append(f"## {block['label']}")
        lines.append(f"_Semantic Scholar query:_ `{block['ss_query']}`  ")
        lines.append(f"_arXiv query:_ `{block['arxiv_query']}`\n")
        if not block["papers"]:
            lines.append("(no results)\n")
            continue
        for p in block["papers"][:8]:
            title = p["title"]
            authors = ", ".join(p["authors"][:3])
            if len(p["authors"]) > 3:
                authors += " et al."
            year = p["year"] or "n.d."
            venue = p["venue"] or "—"
            cites = p["citations"]
            cites_s = f"{cites} citations" if cites else "citations unknown"
            arx = p.get("arxiv_id")
            doi = p.get("doi")
            tags = []
            if arx:
                tags.append(f"[arXiv:{arx}](https://arxiv.org/abs/{arx})")
            if doi:
                tags.append(f"[DOI:{doi}](https://doi.org/{doi})")
            if p.get("local_pdf"):
                tags.append(f"local: `{p['local_pdf']}`")
            tag_s = " · ".join(tags) if tags else ""
            lines.append(f"- **{title}** ({year}). {authors}. _{venue}_, {cites_s}.")
            if tag_s:
                lines.append(f"  {tag_s}")
            abstract = (p.get("abstract") or "").strip()
            if abstract:
                lines.append(f"  > {abstract[:380]}{'…' if len(abstract) > 380 else ''}")
        lines.append("")

    SUMMARY_MD.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote {SUMMARY_MD}")

    total_papers = sum(len(b["papers"]) for b in all_blocks)
    print(
        f"\nDone: {total_papers} unique candidates across {len(all_blocks)} blocks; "
        f"{pdf_count} PDFs downloaded to {PDF_DIR}"
    )


if __name__ == "__main__":
    main()
