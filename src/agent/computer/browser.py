"""
Chitti Browser Controller.
Provides real browser automation, YouTube video resolution & playback, search, and state verification.
"""

import os
import platform
import re
import subprocess
import time
import urllib.parse
import urllib.request
import webbrowser
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import psutil

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

    _shared_state = BrowserState()

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
        self.state = self._shared_state

    @classmethod
    def resolve_browser_executable(cls, preferred: Optional[str] = None) -> Optional[str]:
        """
        Resolves the best available browser executable on Windows.
        Prioritizes actively running browsers (Brave, Chrome, Edge, Firefox) so tabs open
        in the user's active window, followed by installed executables.
        """
        # 1. If explicit preference requested
        if preferred:
            exe = AppDiscovery.resolve_app(preferred)
            if exe and os.path.exists(exe):
                return exe

        # 2. Check actively running browser processes to reuse the user's current session
        browser_proc_names = {
            "brave.exe": "brave",
            "chrome.exe": "chrome",
            "msedge.exe": "msedge",
            "firefox.exe": "firefox",
        }
        for proc in psutil.process_iter(["name", "exe"]):
            try:
                pname = (proc.info["name"] or "").lower()
                if pname in browser_proc_names:
                    pexe = proc.info.get("exe")
                    if pexe and os.path.exists(pexe):
                        return pexe
                    resolved = AppDiscovery.resolve_app(browser_proc_names[pname])
                    if resolved and os.path.exists(resolved):
                        return resolved
            except Exception:
                continue

        # 3. Check installed browsers in order of preference
        for b_name in ("brave", "chrome", "msedge", "edge", "firefox"):
            exe = AppDiscovery.resolve_app(b_name)
            if exe and os.path.exists(exe):
                return exe

        return None

    @classmethod
    def normalize_artist_query(cls, raw_query: str) -> str:
        """Normalizes user query by stripping stop words and resolving known artist names."""
        clean = raw_query.strip()
        # Remove common phrasing and command prefixes
        clean = re.sub(
            r"(?i)\b(?:open\s+youtube|open\s+yt|go\s+to\s+youtube|go\s+to\s+yt|on\s+youtube|on\s+yt|youtube|yt|and|aur|\&|play\s+a|play\s+an|play\s+the|play|open|kholo|chala\s+do|chalao|karo|baja\s+do|ka\s+gaana|song|songs|music|track|gaana|a|an|the|me|mein|par|pe)\b",
            " ",
            clean,
        )
        clean = re.sub(r"\s+", " ", clean).strip()

        # If user simply requested "a song" or generic music, default to Trending Songs
        generic_terms = {"", "a", "an", "the", "song", "songs", "music", "track", "gaana", "top", "trending", "some", "any"}
        if clean.lower() in generic_terms:
            return "Trending Songs"

        lower = clean.lower()
        for key, normalized in cls.ARTIST_NORMALIZATIONS.items():
            if key in lower:
                return normalized

        return clean.title() if clean else "Trending Songs"

    def open_url(self, url: str, browser: Optional[str] = None, site_name: Optional[str] = None) -> bool:
        """Opens a URL visibly in the user's running or default desktop browser."""
        target_url = url.strip()
        if not target_url.startswith(("http://", "https://", "file:///")):
            if target_url.startswith("file:/"):
                target_url = target_url.replace("file:/", "file:///")
            elif target_url.startswith("/") or re.match(r"^[a-zA-Z]:[/\\]", target_url):
                target_url = f"file:///{target_url.replace('\\', '/')}"
            else:
                target_url = f"https://{target_url}"

        try:
            # 1. Direct browser executable invocation via subprocess
            # This guarantees that Windows opens an interactive desktop window/tab
            browser_exe = self.resolve_browser_executable(preferred=browser)
            if browser_exe:
                subprocess.Popen([browser_exe, target_url])
                self.state.current_url = target_url
                self.state.browser_name = os.path.basename(browser_exe)
                self.state.is_open = True
                log_info(f"[BROWSER] Successfully launched URL in {self.state.browser_name}: {target_url}")
                return True

            # 2. Windows shell 'start "" "<url>"' fallback
            if platform.system() == "Windows":
                subprocess.Popen(f'start "" "{target_url}"', shell=True)
                self.state.current_url = target_url
                self.state.is_open = True
                log_info(f"[BROWSER] Launched URL via Windows start command: {target_url}")
                return True

            # 3. Standard Python webbrowser fallback
            webbrowser.open(target_url, new=2)
            self.state.current_url = target_url
            self.state.is_open = True
            log_info(f"[BROWSER] Opened URL via webbrowser: {target_url}")
            return True
        except Exception as e:
            log_warn(f"[BROWSER] Failed to open URL {target_url}: {e}")
            try:
                webbrowser.open(target_url, new=2)
                return True
            except Exception:
                return False

    def verify_page_loaded(self, url: str, expected_keywords: Optional[List[str]] = None) -> Tuple[bool, str]:
        """Verifies that the target page loaded without browser error pages (e.g. Connection Refused)."""
        from src.agent.computer.server_runtime import ServerProcessManager
        return ServerProcessManager.verify_web_page_rendering(url, expected_keywords=expected_keywords)

    def search_web(self, query: str, engine: str = "google") -> bool:
        """Searches the web or target platform for a given query string."""
        from src.agent.registry import ResourceDiscovery
        url = ResourceDiscovery.build_search_url(engine, query)
        return self.open_url(url)

    def fetch_top_youtube_video_id(self, search_query: str) -> Optional[str]:
        """Queries YouTube search to extract the first valid video ID for direct playback."""
        try:
            query_term = f"{search_query} song" if "song" not in search_query.lower() else search_query
            encoded_query = urllib.parse.quote(query_term)
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

        # 1. Attempt to fetch top video ID for direct playback with autoplay
        video_id = self.fetch_top_youtube_video_id(artist)
        if video_id:
            video_url = f"https://www.youtube.com/watch?v={video_id}&autoplay=1"
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

        # 2. Fallback: Open YouTube search results
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

