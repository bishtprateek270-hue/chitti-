"""
Chitti n8n Workflow Automation Client.
Provides outbound webhook triggers, REST API integration, and health checks
for connecting Chitti to local or self-hosted n8n instances.
"""

import json
import logging
import os
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Dict, List, Optional

from src.utils.logging import log_chitti, log_debug, log_error, log_info, log_warning


class N8NClient:
    """Client for triggering n8n webhooks and querying n8n REST APIs."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        webhook_prefix: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: float = 8.0,
    ):
        self.base_url = (base_url or os.getenv("N8N_BASE_URL", "http://localhost:5678")).rstrip("/")
        self.webhook_prefix = (webhook_prefix or os.getenv("N8N_WEBHOOK_PREFIX", f"{self.base_url}/webhook")).rstrip("/")
        self.api_key = api_key or os.getenv("N8N_API_KEY", "").strip() or None
        self.timeout = timeout

    def check_health(self) -> Dict[str, Any]:
        """Checks if the n8n server is reachable and responding."""
        health_urls = [
            f"{self.base_url}/healthz",
            f"{self.base_url}/",
        ]

        last_error = "Unknown connection error"
        for url in health_urls:
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Chitti-AI-Companion/1.0"},
                    method="GET",
                )
                with urllib.request.urlopen(req, timeout=3.0) as resp:
                    status_code = resp.getcode()
                    if 200 <= status_code < 400:
                        return {
                            "status": "online",
                            "reachable": True,
                            "url": self.base_url,
                            "status_code": status_code,
                            "message": f"n8n is running and reachable at {self.base_url}",
                        }
            except Exception as e:
                last_error = str(e)

        return {
            "status": "offline",
            "reachable": False,
            "url": self.base_url,
            "error": last_error,
            "message": f"n8n is not reachable at {self.base_url}. Start n8n using 'npx n8n' or run scripts/run_n8n.bat.",
        }

    def trigger_webhook(
        self,
        endpoint_or_slug: str,
        payload: Optional[Dict[str, Any]] = None,
        method: str = "POST",
        custom_headers: Optional[Dict[str, str]] = None,
    ) -> Dict[str, Any]:
        """
        Triggers an n8n webhook by slug or full URL.
        Example endpoints:
          - 'chitti-task' -> http://localhost:5678/webhook/chitti-task
          - 'chitti-test/webhook' -> http://localhost:5678/webhook/chitti-test/webhook
          - 'http://127.0.0.1:5678/webhook-test/abc' -> used directly
        """
        endpoint = endpoint_or_slug.strip()
        if endpoint.startswith("http://") or endpoint.startswith("https://"):
            full_url = endpoint
        else:
            clean_slug = endpoint.lstrip("/")
            full_url = f"{self.webhook_prefix}/{clean_slug}"

        payload_dict = payload or {}
        # Auto-inject caller metadata
        if "source" not in payload_dict:
            payload_dict["source"] = "chitti_assistant"
        if "user" not in payload_dict:
            payload_dict["user"] = os.getenv("USERNAME", "Prateek")

        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": "Chitti-AI-Companion/1.0",
        }
        if self.api_key:
            headers["X-N8N-API-KEY"] = self.api_key
        if custom_headers:
            headers.update(custom_headers)

        data_bytes = json.dumps(payload_dict, ensure_ascii=False).encode("utf-8") if method.upper() in ("POST", "PUT", "PATCH") else None

        log_chitti(f"[n8n] ⚡ Triggering webhook {method.upper()} -> {full_url}")

        try:
            req = urllib.request.Request(
                full_url,
                data=data_bytes,
                headers=headers,
                method=method.upper(),
            )
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                status_code = resp.getcode()
                raw_body = resp.read().decode("utf-8", errors="replace")
                
                parsed_data = None
                if raw_body.strip():
                    try:
                        parsed_data = json.loads(raw_body)
                    except Exception:
                        parsed_data = raw_body

                log_chitti(f"[n8n] ✓ Webhook executed successfully (Status: {status_code})")
                return {
                    "success": True,
                    "status_code": status_code,
                    "url": full_url,
                    "data": parsed_data,
                    "message": "n8n workflow triggered successfully.",
                }

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            log_warning(f"[n8n] HTTP {e.code} error from webhook {full_url}: {err_body}")
            return {
                "success": False,
                "status_code": e.code,
                "url": full_url,
                "error": err_body or str(e),
                "message": f"n8n webhook returned HTTP error {e.code}: {err_body or str(e)}",
            }
        except urllib.error.URLError as e:
            log_warning(f"[n8n] Connection failed for {full_url}: {e.reason}")
            return {
                "success": False,
                "status_code": 0,
                "url": full_url,
                "error": str(e.reason),
                "message": f"Could not connect to n8n at {self.base_url}. Ensure n8n is running locally or check network.",
            }
        except Exception as e:
            log_error(f"[n8n] Unexpected error triggering webhook: {e}")
            return {
                "success": False,
                "status_code": 0,
                "url": full_url,
                "error": str(e),
                "message": f"Failed to execute n8n webhook: {e}",
            }

    def list_workflows(self) -> Dict[str, Any]:
        """Fetches active workflows via n8n REST API (Requires N8N_API_KEY)."""
        if not self.api_key:
            return {
                "success": False,
                "message": "N8N_API_KEY is not configured in .env. API key is required to list workflows via REST API.",
                "workflows": [],
            }

        url = f"{self.base_url}/api/v1/workflows"
        req = urllib.request.Request(
            url,
            headers={
                "X-N8N-API-KEY": self.api_key,
                "Accept": "application/json",
                "User-Agent": "Chitti-AI-Companion/1.0",
            },
            method="GET",
        )

        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                workflows = data.get("data", [])
                return {
                    "success": True,
                    "count": len(workflows),
                    "workflows": [
                        {
                            "id": w.get("id"),
                            "name": w.get("name"),
                            "active": w.get("active", False),
                        }
                        for w in workflows
                    ],
                    "message": f"Found {len(workflows)} n8n workflow(s).",
                }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"Failed to fetch workflows from n8n API: {e}",
                "workflows": [],
            }
