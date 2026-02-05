import os
import re
import json
import time
import math
import hashlib
from datetime import datetime, timezone, timedelta

import feedparser
import httpx
from dateutil import parser as dtparser
from openai import OpenAI
from openai import APITimeoutError, APIConnectionError, RateLimitError


# ----------------------------
# Config (tweakable via env)
# ----------------------------
MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")  # choose a model you have access to
MAX_ITEMS_PER_FEED = int(os.getenv("MAX_ITEMS_PER_FEED", "50"))
MAX_TOTAL_ITEMS = int(os.getenv("MAX_TOTAL_ITEMS", "400"))
LOOKBACK_DAYS = int(os.getenv("LOOKBACK_DAYS", "7"))

# Keep interests short to reduce prompt size
INTERESTS_MAX_CHARS = int(os.getenv("INTERESTS_MAX_CHARS", "3000"))

# RSS summaries are token hogs—keep small
SUMMARY_MAX_CHARS = int(os.getenv("SUMMARY_MAX_CHARS", "500"))

# Local prefilter reduces model load
PREFILTER_KEEP_TOP = int(os.getenv("PREFILTER_KEEP_TOP", "200"))

# Batch size prevents timeouts
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "50"))

# Threshold for inclusion in digest.md
MIN_SCORE_READ = float(os.getenv("MIN_SCORE_READ", "0.65"))
MAX_RETURNED = int(os.getenv("MAX_RETURNED", "40"))


# ----------------------------
# I/O helpers
# ----------------------------
def load_lines(path: str) -> list[str]:
    with open(path, "r", encoding="utf-8") as f:
        out = []
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            out.append(s)
        return out


def read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


# ----------------------------
# Interests parsing
# ----------------------------
def parse_interests_md(md: str) -> dict:
    """
    Convention:
    - Keywords under heading '## Keywords' (or '# Keywords'), one per line until next heading.
    - Narrative/context is the rest (hard-truncated).
    """
    keywords = []

    m = re.search(r"(?im)^\s*#{1,6}\s+Keywords\s*$", md)
    if m:
        start = m.end()
        rest = md[start:]
        m2 = re.search(r"(?im)^\s*#{1,6}\s+\S", rest)
        block = rest[: m2.start()] if m2 else rest
        for line in block.splitlines():
            line = re.sub(r"^[\-\*\+]\s+", "", line.strip())
            if line:
                keywords.append(line)

    # Keep narrative small
    narrative = md.strip()
    if len(narrative) > INTERESTS_MAX_CHARS:
        narrative = narrative[:INTERESTS_MAX_CHARS] + "…"

    return {"keywords": keywords[:200], "narrative": narrative}


# ----------------------------
# RSS fetching
# ----------------------------
def parse_date(entry) -> datetime | None:
    if getattr(entry, "published_parsed", None):
        return datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
    if getattr(entry, "updated_parsed", None):
        return datetime(*entry.updated_parsed[:6], tzinfo=timezone.utc)

    for key in ("published", "updated", "created"):
        val = entry.get(key)
        if val:
            try:
                dt = dtparser.parse(val)
                return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
            except Exception:
                pass
    return None


def fetch_rss_items(feed_urls: list[str]) -> list[dict]:
    cutoff = datetime.now(timezone.utc) - timedelta(days=LOOKBACK_DAYS)
    items = []

    for url in feed_urls:
        d = feedparser.parse(url)
        source = (d.feed.get("title") or url).strip()

        for e in d.entries[:MAX_ITEMS_PER_FEED]:
            title = (e.get("title") or "").strip()
            link = (e.get("link") or "").strip()
            if not title or not link:
                continue

            dt = parse_date(e)
            if dt and dt < cutoff:
                continue

            summary = (e.get("summary") or e.get("description") or "").strip()
            summary = re.sub(r"\s+", " ", summary)
            if len(summary) > SUMMARY_MAX_CHARS:
                summary = summary[:SUMMARY_MAX_CHARS] + "…"

            items.append(
                {
                    "id": sha1(f"{source}|{title}|{link}"),
                    "source": source,
                    "title": title,
                    "link": link,
                    "published_utc": dt.isoformat() if dt else None,
                    "summary": summary,
                }
            )

    # De-dupe
    dedup = {it["id"]: it for it in items}
    items = list(dedup.values())

    # Newest first
    items.sort(key=lambda x: x["published_utc"] or "", reverse=True)

    return items[:MAX_TOTAL_ITEMS]


# ----------------------------
# Local prefilter
# ----------------------------
def keyword_prefilter(items: list[dict], keywords: list[str], keep_top: int = 200) -> list[dict]:
    kws = [k.lower() for k in keywords if k.strip()]

    def hits(it):
        text = (it.get("title", "") + " " + it.get("summary", "")).lower()
        return sum(1 for k in kws if k in text)

    scored = [(hits(it), it) for it in items]
    matched = [it for s, it in scored if s > 0]

    # If too few match, keep newest N anyway (don’t miss surprises)
    if len(matched) < min(50, keep_top):
        return items[:keep_top]

    matched.sort(key=lambda it: hits(it), reverse=True)
    return matched[:keep_top]


# ----------------------------
# OpenAI client
# ----------------------------
def make_openai_client() -> OpenAI:
    raw = os.environ.get("OPENAI_API_KEY", "")
    key = raw.strip()
    if not key or not key.startswith("sk-"):
        raise RuntimeError("OPENAI_API_KEY is missing or invalid (expected to start with 'sk-').")

    http_client = httpx.Client(
        timeout=httpx.Timeout(connect=30.0, read=300.0, write=30.0, pool=30.0),
        http2=False,
        trust_env=False,
        headers={"Connection": "close", "Accept-Encoding": "gzip"},
    )
    return OpenAI(api_key=key, http_client=http_client)


# ----------------------------
# OpenAI triage (single batch)
# ----------------------------
def call_openai_triage(interests: dict, items: list[dict]) -> dict:
    client = make_openai_client()

    schema = {
        "name": "weekly_toc_digest",
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "week_of": {"type": "string"},
                "notes": {"type": "string"},
                "ranked": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "properties": {
                            "id": {"type": "string"},
                            "title": {"type": "string"},
                            "link": {"type": "string"},
                            "source": {"type": "string"},
                            "published_utc": {"type": ["string", "null"]},
                            "score": {"type": "number"},
                            "why": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["id", "title", "link", "source", "published_utc", "score", "why", "tags"],
                    },
                },
            },
            "required": ["week_of", "notes", "ranked"],
        },
    }

    # Lean items only (don’t send giant blobs)
    lean_items = [
        {
            "id": it["id"],
            "source": it["source"],
            "title": it["title"],
            "link": it["link"],
            "published_utc": it.get("published_utc"),
            "summary": (it.get("summary") or "")[:SUMMARY_MAX_CHARS],
        }
        for it in items
    ]

    prompt = f"""
You are triaging weekly journal table-of-contents RSS items for a researcher.
Use the user's interests below as the basis for relevance.

Output rules:
- Return JSON strictly matching the schema.
- score in [0,1]
- "why": 1–2 concrete sentences grounded in title/summary (no hallucinations)
- "tags": short (e.g., EEG, aperiodic, timescales, HMM, ECG, clinical, state dynamics)
- Rank highest score first.

Interests keywords (high weight):
{json.dumps(interests["keywords"], ensure_ascii=False)}

Interests context (brief):
{interests["narrative"]}

RSS items:
{json.dumps(lean_items, ensure_ascii=False)}
""".strip()

    last_err = None
    for attempt in range(6):
        try:
            resp = client.responses.create(
                model=MODEL,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "weekly_toc_digest",
                        "schema": schema["schema"],
                        "strict": True,
                    }
                },
            )
            return json.loads(resp.output_text)

        except (APITimeoutError, APIConnectionError, RateLimitError) as e:
            last_err = e
            sleep_s = min(60, 2 ** attempt)
            print(f"OpenAI call failed ({type(e).__name__}): {e}. Retrying in {sleep_s}s...")
            time.sleep(sleep_s)

    raise last_err


# ----------------------------
# OpenAI triage (batched)
# ----------------------------
def triage_in_batches(interests: dict, items: list[dict], batch_size: int = 50) -> dict:
    week_of = datetime.now(timezone.utc).date().isoformat()
    total_batches = math.ceil(len(items) / batch_size)

    all_ranked = []
    notes_parts = []

    for i in range(0, len(items), batch_size):
        batch = items[i : i + batch_size]
        print(f"Triage batch {i // batch_size + 1}/{total_batches} ({len(batch)} items)")
        res = call_openai_triage(interests, batch)

        if res.get("notes", "").strip():
            notes_parts.append(res["notes"].strip())

        all_ranked.extend(res.get("ranked", []))

    # De-dupe by id, keep highest score
    best = {}
    for r in all_ranked:
        rid = r["id"]
        if rid not in best or r["score"] > best[rid]["score"]:
            best[rid] = r

    ranked = list(best.values())
    ranked.sort(key=lambda x: x["score"], reverse=True)

    return {
        "week_of": week_of,
        "notes": " ".join(dict.fromkeys(notes_parts))[:1000],
        "ranked": ranked,
    }


# ----------------------------
# Digest rendering
# ----------------------------
def render_digest_md(result: dict, items_by_id: dict[str, dict]) -> str:
    week_of = result["week_of"]
    notes = result.get("notes", "").strip()
    ranked = result.get("ranked", [])

    kept = [r for r in ranked if r["score"] >= MIN_SCORE_READ][:MAX_RETURNED]

    lines = [f"# Weekly ToC Digest (week of {week_of})", ""]
    if notes:
        lines += [notes, ""]

    lines += [
        f"**Included:** {len(kept)} (score ≥ {MIN_SCORE_READ:.2f})  ",
        f"**Scored:** {len(ranked)} total items",
        "",
        "---",
        "",
    ]

    if not kept:
        lines += ["_No items met the relevance threshold this week._", ""]
        return "\n".join(lines)

    for r in kept:
        it = items_by_id.get(r["id"], {})
        title = r["title"]
        link = r["link"]
        source = r["source"]
        score = r["score"]
        why = r["why"].strip()
        tags = ", ".join(r.get("tags", [])) if r.get("tags") else ""
        pub = r.get("published_utc")
        summary = (it.get("summary") or "").strip()

        lines += [
            f"## [{title}]({link})",
            f"*{source}*  ",
            f"Score: **{score:.2f}**" + (f"  \nPublished: {pub}" if pub else ""),
            (f"Tags: {tags}" if tags else ""),
            "",
            why,
            "",
        ]

        if summary:
            lines += [
                "<details>",
                "<summary>RSS summary</summary>",
                "",
                summary,
                "",
                "</details>",
                "",
            ]

        lines += ["---", ""]

    return "\n".join(lines)


# ----------------------------
# Main
# ----------------------------
def main():
    feed_urls = load_lines("feeds.txt")
    interests_md = read_text("interests.md")
    interests = parse_interests_md(interests_md)

    items = fetch_rss_items(feed_urls)
    print(f"Fetched {len(items)} RSS items (pre-filter)")

    if not items:
        today = datetime.now(timezone.utc).date().isoformat()
        md = f"# Weekly ToC Digest (week of {today})\n\n_No RSS items found in the last {LOOKBACK_DAYS} days._\n"
        with open("digest.md", "w", encoding="utf-8") as f:
            f.write(md)
        print("No items; wrote digest.md")
        return

    # Local prefilter reduces prompt size
    items = keyword_prefilter(items, interests["keywords"], keep_top=PREFILTER_KEEP_TOP)
    print(f"Sending {len(items)} RSS items to model (post-filter)")

    items_by_id = {it["id"]: it for it in items}

    # Batched triage prevents timeouts
    result = triage_in_batches(interests, items, batch_size=BATCH_SIZE)

    # Enforce score sorting
    result["ranked"].sort(key=lambda x: x["score"], reverse=True)

    md = render_digest_md(result, items_by_id)

    with open("digest.md", "w", encoding="utf-8") as f:
        f.write(md)

    print("Wrote digest.md")


if __name__ == "__main__":
    main()
