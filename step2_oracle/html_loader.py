"""html_loader.py — Resolve PP samples to their cached HTML + clean it.

The HtmlCache stores one gzipped file per unique URL under
`reboot/data/html_cache/{kp,dp,pp}/<sha>.html.gz`, with `manifest.tsv`
mapping (url, sample_key) → sha256. The cache is published separately
(see `reboot/scripts/fetch_data.sh`) because of its size (~4 GB).
"""

from __future__ import annotations

import csv
import gzip
import re
from pathlib import Path

REBOOT = Path(__file__).resolve().parents[1]
HTML_CACHE_DIR = REBOOT / "data" / "html_cache"
MANIFEST = HTML_CACHE_DIR / "manifest.tsv"

# ── Cleaning regexes (port of oracle_single/html_clean.py) ────────

_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)
_STYLE_BLOCK_RE = re.compile(
    r"<style\b[^>]*>.*?</style>", re.DOTALL | re.IGNORECASE
)
_DATA_URI_RE = re.compile(
    r"data:(?:image|font)/[A-Za-z0-9.+-]+;base64,[A-Za-z0-9+/=]+"
)
_WHITESPACE_RE = re.compile(r"\s+")


def clean_html(html: str, *, max_chars: int = 8000) -> str:
    """Compress HTML for LLM prompt inclusion.

    Removes <style> blocks, HTML comments, and base64 data URIs;
    collapses whitespace; truncates to `max_chars`. Preserves all
    attributes (`style="display:none"`, `action`, etc are forensic).

    Pass `max_chars=0` for no truncation.
    """
    if not html:
        return ""
    s = html
    s = _STYLE_BLOCK_RE.sub(" ", s)
    s = _COMMENT_RE.sub(" ", s)
    s = _DATA_URI_RE.sub("[data:base64...]", s)
    s = _WHITESPACE_RE.sub(" ", s).strip()
    if max_chars > 0 and len(s) > max_chars:
        s = s[:max_chars] + " [HTML_TRUNCATED]"
    return s


# ── Manifest lookup ───────────────────────────────────────────────


class HtmlCacheIndex:
    """In-memory index of the HtmlCache manifest, keyed by sample_key.

    The manifest is a 7-column TSV:
      url, sha256, dataset_tag, size_bytes_compressed, fetched_at_utc,
      orig_filename, sample_key
    """

    def __init__(self) -> None:
        self.by_sample_key: dict[tuple[str, str], str] = {}  # (tag, key) → sha
        self.by_url: dict[str, str] = {}  # url → sha
        if not MANIFEST.exists():
            return
        with MANIFEST.open() as f:
            reader = csv.DictReader(f, delimiter="\t")
            for row in reader:
                sha = (row.get("sha256") or "").strip()
                tag = (row.get("dataset_tag") or "").strip().lower()
                key = (row.get("sample_key") or "").strip()
                url = (row.get("url") or "").strip()
                if sha and tag:
                    if key:
                        self.by_sample_key[(tag, key)] = sha
                    if url:
                        self.by_url[url] = sha

    def find(self, dataset_tag: str, sample_key: str,
             url: str | None = None) -> str | None:
        """Return the sha256 (== filename stem) or None."""
        tag = dataset_tag.lower()
        sha = self.by_sample_key.get((tag, sample_key))
        if sha:
            return sha
        if url:
            return self.by_url.get(url)
        return None

    def html_path(self, dataset_tag: str, sha: str) -> Path:
        return HTML_CACHE_DIR / dataset_tag.lower() / f"{sha}.html.gz"

    def load_html(self, dataset_tag: str, sample_key: str,
                  url: str | None = None) -> str | None:
        sha = self.find(dataset_tag, sample_key, url)
        if not sha:
            return None
        p = self.html_path(dataset_tag, sha)
        if not p.exists():
            return None
        try:
            with gzip.open(p, "rt", encoding="utf-8", errors="ignore") as f:
                return f.read()
        except (OSError, gzip.BadGzipFile):
            return None


# ── Convenience for PP ────────────────────────────────────────────


def load_pp_html(sample_id: int, sample_name: str | None = None,
                 url: str | None = None,
                 index: HtmlCacheIndex | None = None) -> str | None:
    """Resolve a PP sample (id, sample_name, optional url) to raw HTML.

    PP sample_key in the manifest is the original sample_id as a string.
    `sample_name` is `<id>_<host>` — falls back to the id portion if no
    direct match.
    """
    idx = index or HtmlCacheIndex()
    # Try multiple keys: just the id, then the sample_name.
    for key in (str(sample_id), sample_name or ""):
        if not key:
            continue
        html = idx.load_html("pp", key, url=url)
        if html is not None:
            return html
    return None
