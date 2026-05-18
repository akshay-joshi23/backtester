"""Tiny notification adapter.

Sends a short notification line via one of the following, in priority order:
  1. Slack webhook (if SLACK_WEBHOOK_URL is set)
  2. SMTP email (if SMTP_HOST, SMTP_FROM, SMTP_TO are set)
  3. stderr (always — final fallback)

Failures in step 1 or 2 are caught and fall through to stderr; we never
crash the executor over a missing notification channel.
"""

from __future__ import annotations

import json
import logging
import os
import smtplib
import sys
from email.mime.text import MIMEText

logger = logging.getLogger(__name__)


def notify(level: str, title: str, body: str = "") -> None:
    """Send a single notification line. `level` is informational ('info',
    'warn', 'error', 'halt'). Always returns; never raises.
    """
    text = f"[{level.upper()}] {title}"
    if body:
        text += f"\n{body}"
    sent = False
    sent |= _send_slack(text)
    if not sent:
        sent |= _send_email(text, subject=f"[lab] {title}")
    # Always echo to stderr so logs have it too.
    print(text, file=sys.stderr)


def _send_slack(text: str) -> bool:
    url = os.environ.get("SLACK_WEBHOOK_URL")
    if not url:
        return False
    try:
        import urllib.request
        req = urllib.request.Request(
            url, data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5):
            return True
    except Exception as e:
        logger.warning("slack notify failed: %s", e)
        return False


def _send_email(text: str, subject: str) -> bool:
    host = os.environ.get("SMTP_HOST")
    sender = os.environ.get("SMTP_FROM")
    to = os.environ.get("SMTP_TO")
    if not (host and sender and to):
        return False
    port = int(os.environ.get("SMTP_PORT", "587"))
    user = os.environ.get("SMTP_USER")
    password = os.environ.get("SMTP_PASSWORD")
    try:
        msg = MIMEText(text)
        msg["Subject"] = subject
        msg["From"] = sender
        msg["To"] = to
        with smtplib.SMTP(host, port, timeout=10) as smtp:
            smtp.starttls()
            if user and password:
                smtp.login(user, password)
            smtp.send_message(msg)
        return True
    except Exception as e:
        logger.warning("email notify failed: %s", e)
        return False
