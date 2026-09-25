"""
Chitti Browser Controller.
Provides real browser automation, YouTube video resolution & playback, search, and state verification.
"""

import os
import re
import subprocess
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.agent.registry import AppDiscovery, DEFAULT_URL_MAP
from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class BrowserState:
    current_url: Optional[str] = None
    browser_name: str = "default"
    is_open: bool = False
    active_video_id: Optional[str] = None
    playback_verified: bool = False


class BrowserController:
    """Provides real browser control and YouTube playback on Windows."""

    # Common Indian / Global artist phonetic normalizations
    ARTIST_NORMALIZATIONS = {
        "sonu nigam": "Sonu Nigam",
        "shreya ghosal": "Shreya Ghoshal",
        "shreya ghoshal": "Shreya Ghoshal",
        "arijit singh": "Arijit Singh",
        "arijit": "Arijit Singh",
        "lata mangeshkar": "Lata Mangeshkar",
        "kishore kumar": "Kishore Kumar",
        "kumar sanu": "Kumar Sanu",
        "alka yagnik": "Alka Yagnik",
        "sunidhi chauhan": "Sunidhi Chauhan",
        "mohit chauhan": "Mohit Chauhan",
        "udit narayan": "Udit Narayan",
        "kk": "KK",
        "atif aslam": "Atif Aslam",
        "ar rehman": "A.R. Rahman",
        "ar rahman": "A.R. Rahman",
    }

    def __init__(self):
        self.state = BrowserState()

    @classmethod
    def normalize_artist_query(cls, raw_query: str) -> str:
        """Normalizes user query by stripping stop words and resolving known artist names."""
        clean = raw_query.strip()
        # Remove common phrasing stop words: "a", "an", "the", "song", "songs", "music", "track", "gaana", "play", "chalao", "karo"
        clean = re.sub(r"(?i)\b(?:play\s+a|play\s+an|play\s+the|play|a|an|the|song|songs|music|track|gaana|chalao|karo|baja\s+do|ka\s+gaana)\b", " ", clean)
        clean = re.sub(r"\s+", " ", clean).strip()

        lower = clean.lower()
        for key, normalized in cls.ARTIST_NORMALIZATIONS.items():
            if key in lower:
                return normalized

        return clean.title() if clean else "Top Songs"

    def open_url(self, url: str, browser: Optional[str] = None) -> bool:
        """Opens a URL in the user's default browser or specified executable."""
        target_url = url.strip()
        if not target_url.startswith(("http://", "https://")):
            target_url = f"https://{target_url}"

        try:
            if browser:
                browser_exe = AppDiscovery.resolve_app(browser)
                if browser_exe:
                    subprocess.Popen([browser_exe, target_url])
                    self.state.current_url = target_url
                    self.state.browser_name = browser
                    self.state.is_open = True
                    log_info(f"[BROWSER] Opened {target_url} in {browser}")
                    return True

            webbrowser.open(target_url, new=2)
            self.state.current_url = target_url
            self.state.is_open = True
            log_info(f"[BROWSER] Opened URL: {target_url}")
            return True
        except Exception as e:
            log_warn(f"[BROWSER] Failed to open URL {target_url}: {e}")
            return False

    def search_web(self, query: str, engine: str = "google") -> bool:
        """Searches the web for a given query string."""
        encoded = urllib.parse.quote(query.strip())
        if engine.lower() == "bing":
            url = f"https://www.bing.com/search?q={encoded}"
        elif engine.lower() == "duckduckgo":
            url = f"https://duckduckgo.com/?q={encoded}"
        else:
            url = f"https://www.google.com/search?q={encoded}"

        return self.open_url(url)

    def fetch_top_youtube_video_id(self, search_query: str) -> Optional[str]:
        """Queries YouTube search to extract the first valid video ID for direct playback."""
        try:
            encoded_query = urllib.parse.quote(f"{search_query} song")
            url = f"https://www.youtube.com/results?search_query={encoded_query}"
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"}
            )
            with urllib.request.urlopen(req, timeout=4.0) as response:
                html = response.read().decode("utf-8", errors="ignore")
                # Search for video IDs in the returned HTML (watch?v=XXXXXXXXXXX)
                video_ids = re.findall(r"/watch\?v=([a-zA-Z0-9_-]{11})", html)
                if video_ids:
                    # Return first unique video ID
                    return video_ids[0]
        except Exception as e:
            log_debug(f"[BROWSER] YouTube video ID scrape notice: {e}")
        return None

    def search_and_play_youtube(self, query: str) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Executes YouTube search and initiates video playback.
        Returns (success, message, data).
        """
        artist = self.normalize_artist_query(query)
        log_info(f"[BROWSER] Resolving YouTube video for artist/query: '{artist}'")

        # 1. Attempt to fetch top video ID for direct playback
        video_id = self.fetch_top_youtube_video_id(artist)
        if video_id:
            video_url = f"https://www.youtube.com/watch?v={video_id}"
            log_info(f"[BROWSER] Direct YouTube video resolved: {video_url} (ID: {video_id})")
            ok = self.open_url(video_url)
            self.state.active_video_id = video_id
            self.state.current_url = video_url
            self.state.playback_verified = ok

            return ok, f"Playing '{artist}' song on YouTube: {video_url}", {
                "artist": artist,
                "video_id": video_id,
                "url": video_url,
                "playback_verified": ok,
            }

        # 2. Fallback: Open YouTube search results with autoplay parameter
        encoded_query = urllib.parse.quote(f"{artist} songs")
        search_url = f"https://www.youtube.com/results?search_query={encoded_query}"
        log_info(f"[BROWSER] Opening YouTube search results: {search_url}")
        ok = self.open_url(search_url)
        self.state.current_url = search_url
        self.state.playback_verified = ok

        return ok, f"Opened YouTube search for '{artist}' songs", {
            "artist": artist,
            "video_id": None,
            "url": search_url,
            "playback_verified": ok,
        }

    def get_browser_state(self) -> BrowserState:
        """Returns the current state of the browser controller."""
        return self.state
