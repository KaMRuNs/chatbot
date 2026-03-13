"""
Alarm / Reminder Tool — Sets a timed reminder for the Action Agent.

Uses a background thread to wait the specified duration, then writes the
reminder to a shared state that the Streamlit UI polls to display a toast
notification.
"""

import threading
import json
import os
from datetime import datetime, timedelta
from langchain_core.tools import tool

# Shared file where pending alarms are written so Streamlit can pick them up
ALARM_FILE = os.path.join(os.path.dirname(__file__), "..", "pending_alarms.json")


def _write_alarm(alarm_data: dict):
    """Writes a triggered alarm to the shared alarm file for the UI to read."""
    alarms = []
    if os.path.exists(ALARM_FILE):
        with open(ALARM_FILE, "r", encoding="utf-8") as f:
            try:
                alarms = json.load(f)
            except json.JSONDecodeError:
                alarms = []

    alarms.append(alarm_data)

    with open(ALARM_FILE, "w", encoding="utf-8") as f:
        json.dump(alarms, f, indent=2, ensure_ascii=False)


def _alarm_callback(message: str, trigger_time: str):
    """Called by the timer thread when the alarm fires."""
    _write_alarm({
        "message": message,
        "trigger_time": trigger_time,
        "fired_at": datetime.now().isoformat(),
        "status": "fired",
    })


@tool
def set_alarm(seconds: int, message: str) -> str:
    """Set a reminder/alarm that will trigger after a given number of seconds.

    Args:
        seconds: Number of seconds from now until the alarm triggers (e.g. 60 for 1 minute).
        message: The reminder message to display when the alarm goes off (e.g. "Time to take a break!").
    """
    if seconds <= 0:
        return "Error: seconds must be a positive number."
    if seconds > 86400:
        return "Error: maximum alarm duration is 24 hours (86400 seconds)."

    trigger_time = (datetime.now() + timedelta(seconds=seconds)).isoformat()

    timer = threading.Timer(seconds, _alarm_callback, args=[message, trigger_time])
    timer.daemon = True
    timer.start()

    # Format a human-readable time
    if seconds < 60:
        time_str = f"{seconds} second(s)"
    elif seconds < 3600:
        mins = seconds // 60
        secs = seconds % 60
        time_str = f"{mins} minute(s)" + (f" {secs} second(s)" if secs else "")
    else:
        hrs = seconds // 3600
        mins = (seconds % 3600) // 60
        time_str = f"{hrs} hour(s)" + (f" {mins} minute(s)" if mins else "")

    return (
        f"Alarm set successfully!\n"
        f"  Message: {message}\n"
        f"  Will trigger in: {time_str}\n"
        f"  Trigger time: {trigger_time}"
    )
