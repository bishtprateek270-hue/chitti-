"""
Chitti News & Current Affairs Service.
Fetches top news headlines using official NewsAPI.org (if configured)
with seamless automatic fallback to real-time Google News RSS feeds (100% free, zero-key, unlimited).
Formats clean, spoken-friendly summaries for Chitti's TTS.
"""

import json
import logging
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from typing import Any, Dict, List, Optional

from src.utils.logging import log_chitti, log_debug, log_error, log_info, log_warning


TOPIC_FEED_MAP = {
    "technology": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
    "tech": "https://news.google.com/rss/headlines/section/topic/TECHNOLOGY?hl=en-IN&gl=IN&ceid=IN:en",
    "business": "https://news.google.com/rss/headlines/section/topic/BUSINESS?hl=en-IN&gl=IN&ceid=IN:en",
    "science": "https://news.google.com/rss/headlines/section/topic/SCIENCE?hl=en-IN&gl=IN&ceid=IN:en",
    "sports": "https://news.google.com/rss/headlines/section/topic/SPORTS?hl=en-IN&gl=IN&ceid=IN:en",
    "general": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "india": "https://news.google.com/rss?hl=en-IN&gl=IN&ceid=IN:en",
    "world": "https://news.google.com/rss/headlines/section/topic/WORLD?hl=en-IN&gl=IN&ceid=IN:en",
}


class NewsService:
    """Multi-source News Service for real-time headlines and spoken digests."""

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("NEWS_API_KEY", "").strip() or None
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_ttl = 600.0  # 10 minutes cache

    def fetch_headlines(self, category: str = "technology", limit: int = 4) -> List[Dict[str, str]]:
        """
        Fetches headlines from NewsAPI.org or Google News RSS.
        Returns a list of dicts: [{"title": str, "source": str, "link": str}]
        """
        norm_cat = category.strip().lower()
        cache_key = f"{norm_cat}_{limit}"
        now = time.time()

        if cache_key in self._cache:
            entry = self._cache[cache_key]
            if now - entry["timestamp"] < self._cache_ttl:
                return entry["data"]

        results = []

        # 1. Try NewsAPI.org if API key is provided
        if self.api_key:
            results = self._fetch_newsapi(norm_cat, limit)

        # 2. Seamless fallback to Google News RSS
        if not results:
            results = self._fetch_google_rss(norm_cat, limit)

        if results:
            self._cache[cache_key] = {"data": results, "timestamp": now}

        return results

    def _fetch_newsapi(self, category: str, limit: int) -> List[Dict[str, str]]:
        """Fetches from newsapi.org."""
        cat = category if category in ("business", "entertainment", "general", "health", "science", "sports", "technology") else "technology"
        endpoints = [
            f"https://newsapi.org/v2/top-headlines?country=in&category={cat}&pageSize={limit}&apiKey={self.api_key}",
            f"https://newsapi.org/v2/top-headlines?category={cat}&pageSize={limit}&apiKey={self.api_key}",
            f"https://newsapi.org/v2/everything?q={cat}&sortBy=publishedAt&pageSize={limit}&apiKey={self.api_key}",
        ]

        for url in endpoints:
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "Chitti-AI/1.0"})
                with urllib.request.urlopen(req, timeout=5.0) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    if data.get("status") == "ok":
                        articles = data.get("articles", [])
                        out = []
                        for art in articles[:limit]:
                            title = art.get("title") or ""
                            # Strip trailing source e.g. "Headline - Source Name"
                            clean_title = re.sub(r"\s+-\s+[^-]+$", "", title).strip()
                            source = (art.get("source") or {}).get("name") or "News"
                            if clean_title and clean_title != "[Removed]":
                                out.append({
                                    "title": clean_title,
                                    "source": source,
                                    "link": art.get("url") or "",
                                })
                        if out:
                            return out
            except Exception as e:
                log_debug(f"[NewsService] NewsAPI endpoint {url} notice: {e}")

        return []

    def _fetch_google_rss(self, category: str, limit: int) -> List[Dict[str, str]]:
        """Fetches from Google News RSS feeds (Keyless & Unlimited)."""
        feed_url = TOPIC_FEED_MAP.get(category, TOPIC_FEED_MAP["technology"])
        try:
            req = urllib.request.Request(
                feed_url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
            )
            with urllib.request.urlopen(req, timeout=6.0) as resp:
                xml_data = resp.read()
                root = ET.fromstring(xml_data)

                items = root.findall(".//item")
                out = []
                for item in items[:limit]:
                    raw_title = item.find("title")
                    title_text = raw_title.text if raw_title is not None and raw_title.text else ""
                    # Google News titles usually format as: "Title - Source"
                    m = re.match(r"^(.*?)\s+-\s+([^-]+)$", title_text)
                    if m:
                        title = m.group(1).strip()
                        source = m.group(2).strip()
                    else:
                        title = title_text.strip()
                        source = "Google News"

                    link_elem = item.find("link")
                    link = link_elem.text if link_elem is not None and link_elem.text else ""

                    if title:
                        out.append({
                            "title": title,
                            "source": source,
                            "link": link,
                        })
                return out
        except Exception as e:
            log_warning(f"[NewsService] Google News RSS fetch notice: {e}")
            return []

    def format_spoken_summary(
        self,
        headlines: List[Dict[str, str]],
        category: str = "technology",
        lang: str = "en",
    ) -> str:
        """Formats headlines into natural spoken text for Chitti TTS."""
        if not headlines:
            if lang in ("hi", "hinglish"):
                return "Maaf kijiye, abhi news headlines fetch nahi ho paayi. Internet connection check karein."
            return "I couldn't fetch the latest headlines right now. Please check your internet connection."

        cat_label = category.capitalize() if category else "Today's"
        ordinals_en = ["First", "Second", "Third", "Fourth", "Finally"]
        ordinals_hi = ["Pehle", "Doosri khabar", "Teesri khabar", "Chauthi khabar", "Aakhri khabar"]

        is_hindi = lang in ("hi", "hinglish")
        lines = []

        if is_hindi:
            lines.append(f"Yeh hain aaj ki top {category} headlines:")
            for i, item in enumerate(headlines):
                ord_word = ordinals_hi[i] if i < len(ordinals_hi) else f"Number {i+1}"
                lines.append(f"{ord_word}: {item['title']}.")
        else:
            lines.append(f"Here are the top {category} headlines today:")
            for i, item in enumerate(headlines):
                ord_word = ordinals_en[i] if i < len(ordinals_en) else f"Item {i+1}"
                lines.append(f"{ord_word}: {item['title']}.")

        return " ".join(lines)
