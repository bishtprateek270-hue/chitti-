"""
Chitti Browser Controller.
Controls browser navigation, tab management, web searches, and URL visits on Windows.
"""

import os
import subprocess
import urllib.parse
import webbrowser
from dataclasses import dataclass
from typing import Optional

from src.agent.registry import AppDiscovery, DEFAULT_URL_MAP
from src.utils.logging import log_debug, log_info, log_warn


@dataclass
class BrowserState:
    current_url: Optional[str] = None
    browser_name: str = "default"
    is_open: bool = False


class BrowserController:
    """Provides high-level browser control on Windows."""

    def __init__(self):
        self.state = BrowserState()

    def open_url(self, url: str, browser: Optional[str] = None) -> bool:
        """Opens a URL in the specified browser or system default browser."""
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
                    log_info(f"Opened URL in {browser}: {target_url}")
                    return True

            # Standard system default browser open
            webbrowser.open(target_url, new=2)
            self.state.current_url = target_url
            self.state.is_open = True
            log_info(f"Opened URL in default browser: {target_url}")
            return True
        except Exception as e:
            log_warn(f"Failed to open browser for URL {target_url}: {e}")
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

    def open_site(self, site_name: str) -> bool:
        """Opens a well-known site by alias (e.g. 'github', 'youtube', 'chatgpt')."""
        clean_name = site_name.strip().lower()
        if clean_name in DEFAULT_URL_MAP:
            return self.open_url(DEFAULT_URL_MAP[clean_name])
        return self.search_web(site_name)
