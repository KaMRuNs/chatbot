"""
Calendar Tool — Generates downloadable .ics calendar event files.
Compatible with Google Calendar, Outlook, and Apple Calendar.
No API keys required.
"""

import os
import tempfile
from datetime import datetime, timedelta
from icalendar import Calendar, Event
from langchain_core.tools import tool


@tool
def create_calendar_event(
    title: str,
    date: str,
    time: str = "09:00",
    duration_minutes: int = 60,
    description: str = "",
    location: str = ""
) -> str:
    """
    Create a calendar event and save it as a downloadable .ics file.
    
    Use this tool ONLY when the user asks to explicitly schedule an event:
    - "Schedule a meeting", "Add an event to my calendar"
    - "Book a slot", "Set up a calendar event"
    
    Args:
        title: The title/name of the event (e.g., "Job Interview at Google").
        date: The date of the event in YYYY-MM-DD format (e.g., "2026-03-18").
        time: The start time in HH:MM 24-hour format (e.g., "14:30"). Defaults to 09:00.
        duration_minutes: Duration of the event in minutes. Defaults to 60.
        description: Optional description or notes for the event.
        location: Optional location (physical address or "Google Meet" etc).
    
    Returns:
        Path to the generated .ics file which the user can download and import.
    """
    try:
        # Parse datetime
        start_dt = datetime.strptime(f"{date} {time}", "%Y-%m-%d %H:%M")
        end_dt = start_dt + timedelta(minutes=duration_minutes)

        # Build iCalendar object
        cal = Calendar()
        cal.add("prodid", "-//CareerPilot//EN")
        cal.add("version", "2.0")
        cal.add("calscale", "GREGORIAN")
        cal.add("method", "PUBLISH")

        event = Event()
        event.add("summary", title)
        event.add("dtstart", start_dt)
        event.add("dtend", end_dt)
        event.add("dtstamp", datetime.now())
        if description:
            event.add("description", description)
        if location:
            event.add("location", location)

        cal.add_component(event)

        # Save to temp file
        safe_title = "".join(c for c in title if c.isalnum() or c in " _-").strip().replace(" ", "_")[:40]
        tmp_path = os.path.join(tempfile.gettempdir(), f"{safe_title}.ics")
        with open(tmp_path, "wb") as f:
            f.write(cal.to_ical())

        summary = (
            f"✅ Calendar event created!\n\n"
            f"**Title:** {title}\n"
            f"**Date:** {start_dt.strftime('%B %d, %Y')}\n"
            f"**Time:** {start_dt.strftime('%I:%M %p')} – {end_dt.strftime('%I:%M %p')}\n"
        )
        if location:
            summary += f"**Location:** {location}\n"
        if description:
            summary += f"**Notes:** {description}\n"

        summary += f"\n📁 ICS_FILE_PATH:{tmp_path}"
        return summary
    except ValueError as e:
        return (
            f"❌ Could not parse the date/time. Please use YYYY-MM-DD for date and HH:MM for time.\n"
            f"Error: {e}"
        )
    except Exception as e:
        return f"❌ Failed to create calendar event: {e}"
