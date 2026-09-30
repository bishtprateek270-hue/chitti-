"""
Chitti Messaging & Communication Engine.
Provides robust local dispatch for WhatsApp and Gmail/Email using:
1. Native SMTP (smtplib + email.mime) with Google App Passwords / Environment credentials for instant background email dispatch.
2. Direct pre-filled web URLs for Gmail and WhatsApp Web (no reliance on fragile DOM scraping or paid cloud APIs).
3. Browser UI automation fallbacks for active user sessions.
"""

import email.message
import json
import os
import re
import smtplib
import time
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from src.utils.logging import log_chitti, log_debug, log_info, log_warn
from src.utils.text import clean_contact_query, strip_emojis


@dataclass
class DispatchResult:
    success: bool
    message: str
    evidence: Optional[str] = None
    channel: str = "generic"
    method: str = "unknown"


class EmailDispatcher:
    """Handles email composition and delivery via direct SMTP or pre-filled Gmail Web compose."""

    @classmethod
    def get_smtp_credentials(cls) -> Tuple[Optional[str], Optional[str]]:
        """Retrieves user's email address and app password from environment or config file."""
        email_addr = os.getenv("CHITTI_EMAIL_ADDRESS") or os.getenv("GMAIL_USER") or os.getenv("EMAIL_USER")
        email_pass = os.getenv("CHITTI_EMAIL_APP_PASSWORD") or os.getenv("GMAIL_APP_PASSWORD") or os.getenv("EMAIL_PASS")

        if email_addr and email_pass:
            return email_addr.strip(), email_pass.strip()

        # Check config files
        config_paths = [
            Path("data/config.json"),
            Path("config.json"),
            Path("src/config.json"),
        ]
        try:
            config_paths.append(Path.home() / ".chitti" / "config.json")
        except Exception:
            pass

        for cfg_path in config_paths:
            try:
                if cfg_path.exists():
                    with open(cfg_path, "r", encoding="utf-8") as f:
                        data = json.load(f)
                        addr = data.get("email_address") or data.get("gmail_user") or (data.get("email", {}) if isinstance(data.get("email"), dict) else {}).get("address")
                        pwd = data.get("email_app_password") or data.get("gmail_app_password") or (data.get("email", {}) if isinstance(data.get("email"), dict) else {}).get("app_password")
                        if addr and pwd:
                            return str(addr).strip(), str(pwd).strip()
            except Exception as e:
                log_debug(f"[EMAIL] Error reading config from {cfg_path}: {e}")

        return None, None

    @classmethod
    def send_email_smtp(
        cls,
        recipient: str,
        subject: str,
        body: str,
        sender_email: Optional[str] = None,
        sender_password: Optional[str] = None,
        smtp_server: str = "smtp.gmail.com",
        smtp_port: int = 587,
    ) -> DispatchResult:
        """Sends an email directly via SMTP using Python's standard library (free, zero external API keys)."""
        if not sender_email or not sender_password:
            sender_email, sender_password = cls.get_smtp_credentials()

        if not sender_email or not sender_password:
            return DispatchResult(
                success=False,
                message="SMTP credentials not configured. Falling back to browser Gmail workflow.",
                evidence="NO_CREDENTIALS",
                channel="Email",
                method="smtp",
            )

        try:
            log_info(f"[EMAIL] Connecting to SMTP server {smtp_server}:{smtp_port} for {sender_email}...")
            msg = email.message.EmailMessage()
            msg["From"] = sender_email
            msg["To"] = recipient
            msg["Subject"] = subject
            msg.set_content(body)

            with smtplib.SMTP(smtp_server, smtp_port, timeout=12) as server:
                server.ehlo()
                server.starttls()
                server.ehlo()
                server.login(sender_email, sender_password)
                server.send_message(msg)

            log_chitti(f"[CHITTI] [EMAIL] Direct SMTP dispatch successful to {recipient}")
            return DispatchResult(
                success=True,
                message=f"Email successfully dispatched to {recipient} with subject '{subject}'.",
                evidence=f"SMTP confirmation: Sent from {sender_email} to {recipient}",
                channel="Email",
                method="smtp",
            )
        except Exception as e:
            log_warn(f"[EMAIL] SMTP dispatch error: {e}")
            return DispatchResult(
                success=False,
                message=f"SMTP dispatch failed: {e}",
                evidence=str(e),
                channel="Email",
                method="smtp",
            )

    @classmethod
    def get_gmail_compose_url(cls, recipient: str, subject: str, body: str) -> str:
        """Generates a direct pre-filled Gmail Web compose URL."""
        to_q = urllib.parse.quote(recipient)
        su_q = urllib.parse.quote(subject)
        body_q = urllib.parse.quote(body)
        return f"https://mail.google.com/mail/u/0/?view=cm&fs=1&to={to_q}&su={su_q}&body={body_q}"


class WhatsAppDispatcher:
    """Handles WhatsApp message dispatch via direct web chat links or automated search."""

    @classmethod
    def is_phone_number(cls, target: str) -> bool:
        """Determines if the target string is a phone number."""
        clean = re.sub(r"[\s\-\(\)\+]", "", target)
        return len(clean) >= 7 and clean.isdigit()

    @classmethod
    def normalize_phone_number(cls, phone: str, default_country_code: str = "91") -> str:
        """Normalizes a phone number to standard international format without '+'."""
        clean = re.sub(r"[\s\-\(\)]", "", phone.strip())
        if clean.startswith("+"):
            clean = clean[1:]
        elif len(clean) == 10 and clean.isdigit():
            # Add default country code (e.g. 91 for India) if 10 digits
            clean = f"{default_country_code}{clean}"
        return clean

    @classmethod
    def get_whatsapp_url(cls, contact_or_phone: str, message: str) -> str:
        """
        Generates direct WhatsApp Web link.
        If a phone number is provided, returns direct send URL with pre-filled text.
        """
        msg_q = urllib.parse.quote(message)
        if cls.is_phone_number(contact_or_phone):
            phone = cls.normalize_phone_number(contact_or_phone)
            return f"https://web.whatsapp.com/send?phone={phone}&text={msg_q}"
        return "https://web.whatsapp.com"

    @classmethod
    def send_via_pywhatkit(cls, phone: str, message: str) -> DispatchResult:
        """Attempts dispatch via pywhatkit if installed on the system."""
        try:
            import pywhatkit
            norm_phone = "+" + cls.normalize_phone_number(phone)
            log_info(f"[WHATSAPP] Dispatching message to {norm_phone} via pywhatkit...")
            pywhatkit.sendwhatmsg_instantly(norm_phone, message, wait_time=10, tab_close=True, close_time=3)
            return DispatchResult(
                success=True,
                message=f"Message sent to {phone} via WhatsApp automation.",
                evidence=f"pywhatkit dispatched to {norm_phone}",
                channel="WhatsApp",
                method="pywhatkit",
            )
        except ImportError:
            return DispatchResult(
                success=False,
                message="pywhatkit not installed.",
                evidence="MODULE_NOT_FOUND",
                channel="WhatsApp",
                method="pywhatkit",
            )
        except Exception as e:
            return DispatchResult(
                success=False,
                message=f"pywhatkit dispatch failed: {e}",
                evidence=str(e),
                channel="WhatsApp",
                method="pywhatkit",
            )
