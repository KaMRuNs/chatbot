"""
Email Tool — Simulated email sending for the Action Agent.

Currently runs in simulation mode: it logs the email details instead of
actually sending. To upgrade to real SMTP, replace the body of _send_real_email
and set SIMULATE_EMAIL=false in your .env file.
"""

import os
import json
import smtplib
from email.mime.text import MIMEText
from datetime import datetime
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# Where simulated emails are saved
EMAIL_LOG_FILE = os.path.join(os.path.dirname(__file__), "..", "email_log.json")

SIMULATE = os.getenv("SIMULATE_EMAIL", "true").lower() == "true"


def _log_email(to: str, subject: str, body: str, status: str):
    """Appends the email details to a local JSON log file."""
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "to": to,
        "subject": subject,
        "body": body,
        "status": status,
    }

    logs = []
    if os.path.exists(EMAIL_LOG_FILE):
        with open(EMAIL_LOG_FILE, "r", encoding="utf-8") as f:
            try:
                logs = json.load(f)
            except json.JSONDecodeError:
                logs = []

    logs.append(log_entry)

    with open(EMAIL_LOG_FILE, "w", encoding="utf-8") as f:
        json.dump(logs, f, indent=2, ensure_ascii=False)


def _send_real_email(to: str, subject: str, body: str):
    """Sends an actual email via SMTP. Requires EMAIL_ADDRESS and EMAIL_APP_PASSWORD in .env."""
    sender = os.getenv("EMAIL_ADDRESS")
    password = os.getenv("EMAIL_APP_PASSWORD")
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

    if not sender or not password:
        raise ValueError(
            "EMAIL_ADDRESS and EMAIL_APP_PASSWORD must be set in .env to send real emails."
        )

    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)


@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a recipient.

    Args:
        to: The recipient's email address (e.g. "user@example.com").
        subject: The subject line of the email.
        body: The body text of the email.
    """
    try:
        if SIMULATE:
            _log_email(to, subject, body, status="simulated")
            return (
                f"[SIMULATED] Email composed successfully!\n"
                f"  To: {to}\n"
                f"  Subject: {subject}\n"
                f"  Body: {body}\n"
                f"(Logged to email_log.json. Set SIMULATE_EMAIL=false in .env to send real emails.)"
            )
        else:
            _send_real_email(to, subject, body)
            _log_email(to, subject, body, status="sent")
            return f"Email sent successfully to {to} with subject '{subject}'."
    except Exception as e:
        _log_email(to, subject, body, status=f"error: {e}")
        return f"Failed to send email: {e}"
