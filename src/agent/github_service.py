"""
Chitti GitHub Integration Service.
Provides repository queries, pull request tracking, commit logs, and issue management
using the GitHub REST API v3.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from src.utils.logging import log_chitti, log_debug, log_error, log_info, log_warning


class GitHubService:
    """GitHub client for managing repositories, PRs, issues, and workflow runs."""

    def __init__(self, token: Optional[str] = None, username: Optional[str] = None):
        self.token = token or os.getenv("GITHUB_TOKEN", "").strip() or None
        self.username = username or os.getenv("GITHUB_USERNAME", "bishtprateek270-hue").strip()
        self.base_url = "https://api.github.com"

    def _make_request(self, endpoint: str, method: str = "GET", data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Makes an authenticated request to GitHub REST API."""
        if not self.token:
            return {"success": False, "error": "GITHUB_TOKEN not configured in .env."}

        url = f"{self.base_url}{endpoint}" if endpoint.startswith("/") else f"{self.base_url}/{endpoint}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "Chitti-AI-Assistant/1.0",
            "X-GitHub-Api-Version": "2022-11-28",
        }

        body_bytes = json.dumps(data).encode("utf-8") if data else None

        try:
            req = urllib.request.Request(url, data=body_bytes, headers=headers, method=method)
            with urllib.request.urlopen(req, timeout=8.0) as resp:
                status_code = resp.getcode()
                raw = resp.read().decode("utf-8")
                parsed = json.loads(raw) if raw.strip() else {}
                return {"success": True, "status_code": status_code, "data": parsed}
        except urllib.error.HTTPError as e:
            err_msg = e.read().decode("utf-8", errors="replace")
            log_warning(f"[GitHubService] HTTP Error {e.code} on {endpoint}: {err_msg}")
            return {"success": False, "status_code": e.code, "error": err_msg}
        except Exception as e:
            log_error(f"[GitHubService] Error on {endpoint}: {e}")
            return {"success": False, "error": str(e)}

    def get_authenticated_user(self) -> Dict[str, Any]:
        """Fetches the profile info of the authenticated user."""
        res = self._make_request("/user")
        if res.get("success"):
            data = res.get("data", {})
            return {
                "success": True,
                "login": data.get("login"),
                "name": data.get("name"),
                "public_repos": data.get("public_repos"),
                "followers": data.get("followers"),
                "html_url": data.get("html_url"),
            }
        return res

    def list_repositories(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Lists repositories for the authenticated user."""
        res = self._make_request(f"/user/repos?sort=updated&per_page={limit}")
        if res.get("success"):
            repos = res.get("data", [])
            return [
                {
                    "name": r.get("name"),
                    "full_name": r.get("full_name"),
                    "private": r.get("private"),
                    "html_url": r.get("html_url"),
                    "stars": r.get("stargazers_count", 0),
                    "description": r.get("description") or "",
                }
                for r in repos
            ]
        return []

    def get_repo_pull_requests(self, repo_name: str, state: str = "open") -> List[Dict[str, Any]]:
        """Fetches pull requests for a given repository."""
        full_repo = repo_name if "/" in repo_name else f"{self.username}/{repo_name}"
        res = self._make_request(f"/repos/{full_repo}/pulls?state={state}")
        if res.get("success"):
            prs = res.get("data", [])
            return [
                {
                    "number": pr.get("number"),
                    "title": pr.get("title"),
                    "state": pr.get("state"),
                    "user": (pr.get("user") or {}).get("login"),
                    "created_at": pr.get("created_at"),
                    "html_url": pr.get("html_url"),
                }
                for pr in prs
            ]
        return []

    def create_issue(self, repo_name: str, title: str, body: str = "") -> Dict[str, Any]:
        """Creates a new issue in a repository."""
        full_repo = repo_name if "/" in repo_name else f"{self.username}/{repo_name}"
        payload = {"title": title, "body": body}
        res = self._make_request(f"/repos/{full_repo}/issues", method="POST", data=payload)
        if res.get("success"):
            issue = res.get("data", {})
            return {
                "success": True,
                "number": issue.get("number"),
                "title": issue.get("title"),
                "html_url": issue.get("html_url"),
            }
        return res
