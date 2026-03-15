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
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
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


def _send_real_email(to: str, subject: str, body: str, attachments: list[str] = None, sender: str = None, password: str = None):
    """Sends an actual email via SMTP with optional attachments."""
    sender = sender or os.getenv("EMAIL_ADDRESS")
    password = password or os.getenv("EMAIL_APP_PASSWORD")
    smtp_server = os.getenv("EMAIL_SMTP_SERVER", "smtp.gmail.com")
    smtp_port = int(os.getenv("EMAIL_SMTP_PORT", "587"))

    if not sender or not password:
        raise ValueError(
            "No email credentials found. Please enter your Gmail address and App Password in the sidebar."
        )

    msg = MIMEMultipart()
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to
    msg.attach(MIMEText(body, "plain"))

    if attachments:
        for file_path in attachments:
            if os.path.exists(file_path):
                filename = os.path.basename(file_path)
                with open(file_path, "rb") as f:
                    part = MIMEBase("application", "octet-stream")
                    part.set_payload(f.read())
                    encoders.encode_base64(part)
                    part.add_header("Content-Disposition", f"attachment; filename={filename}")
                    msg.attach(part)

    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender, password)
        server.send_message(msg)


# A thread-safe context for passing session credentials into tools
_email_context: dict = {}

def set_email_context(sender: str, password: str):
    """Called from app.py to inject user-provided credentials before tool use."""
    _email_context["sender"] = sender
    _email_context["password"] = password

def get_email_context():
    return _email_context.get("sender", ""), _email_context.get("password", "")


@tool
def send_email(to: str, subject: str, body: str, attachments: list[str] = None) -> str:
    """Send an email to a recipient with optional file attachments.

    Args:
        to: The recipient's email address (e.g. "user@example.com").
        subject: The subject line of the email.
        body: The body text of the email.
        attachments: An optional list of absolute paths to files to attach (e.g. ["/path/to/resume.pdf"]).
    """
    try:
        s, p = get_email_context()
        # Send for real if credentials available (either via sidebar or .env)
        can_send = bool(s or os.getenv("EMAIL_ADDRESS"))
        
        if not can_send:
            _log_email(to, subject, body, status="simulated")
            msg = (
                f"[SIMULATED] Email composed!\n"
                f"  To: {to}\n"
                f"  Subject: {subject}\n"
                f"  Body preview: {body[:300]}...\n"
            )
            if attachments:
                msg += f"  Attachments: {', '.join([os.path.basename(a) for a in attachments])}\n"
            msg += f"\n💡 Enter your Gmail & App Password in the **sidebar → 📧 Email Settings** to send real emails."
            return msg
        else:
            s, p = get_email_context()
            _send_real_email(to, subject, body, attachments=attachments, sender=s or None, password=p or None)
            _log_email(to, subject, body, status="sent")
            msg = f"Email sent successfully to {to} with subject '{subject}'."
            if attachments:
                msg += f" (Attached: {', '.join([os.path.basename(a) for a in attachments])})"
            return msg
    except Exception as e:
        _log_email(to, subject, body, status=f"error: {e}")
        return f"Failed to send email: {e}"
