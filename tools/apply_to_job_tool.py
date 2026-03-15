"""
Apply to Job Tool — A high-level orchestration tool that:
1. Searches for relevant jobs using SerpApi Google Jobs.
2. Matches user skills against the first job listing.
3. Generates a personalized cover letter.
4. Sends an application email to a provided address (or flags that no email was found).
"""

import os
from langchain_core.tools import tool
from serpapi import GoogleSearch
from dotenv import load_dotenv
from utils.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv(override=True)


def _search_jobs_internal(query: str, location: str = "") -> list:
    """Internal helper to search Google Jobs and return raw jobs list."""
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        return []
    params = {"engine": "google_jobs", "q": query, "api_key": api_key}
    if location:
        params["location"] = location
    try:
        results = GoogleSearch(params).get_dict()
        return results.get("jobs_results", [])[:5]
    except Exception:
        return []


def _generate_cover_letter_internal(job_description: str, resume_text: str) -> str:
    """Internal helper to generate cover letter using LLM."""
    llm = get_llm()
    system_prompt = (
        "You are an expert career coach. Write a concise, compelling, and professional cover letter "
        "based on the resume and job description provided. Leave [Company Name] as a placeholder if not given."
    )
    user_prompt = (
        f"--- JOB DESCRIPTION ---\n{job_description}\n\n"
        f"--- RESUME ---\n{resume_text}\n\n"
        "Write the cover letter:"
    )
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content
    except Exception as e:
        return f"Failed to generate cover letter: {e}"


def _send_email_internal(to: str, subject: str, body: str) -> str:
    """Internal helper to send email using smtplib, reading credentials from context."""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    from tools.email_tool import get_email_context

    smtp_user, smtp_password = get_email_context()

    # Fall back to .env if not set via UI
    if not smtp_user:
        smtp_user = os.getenv("EMAIL_ADDRESS")
    if not smtp_password:
        smtp_password = os.getenv("EMAIL_APP_PASSWORD")

    if not smtp_user or not smtp_password:
        return (
            "⚠️ No email credentials found. Please enter your **Gmail address** and "
            "**App Password** in the sidebar under **📧 Email Settings** to enable real email sending."
        )

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = smtp_user
        msg["To"] = to
        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as server:
            server.login(smtp_user, smtp_password)
            server.sendmail(smtp_user, to, msg.as_string())

        return f"✅ Application email successfully sent to {to}!"
    except Exception as e:
        return f"❌ Failed to send email: {e}"


@tool
def apply_to_job(
    job_query: str,
    resume_text: str,
    recipient_email: str = "",
    location: str = ""
) -> str:
    """
    Automatically apply to a job by:
    1. Searching for job listings matching the query.
    2. Identifying the best matching role.
    3. Generating a personalized cover letter based on the user's resume.
    4. Sending the application email to the provided recipient (if given).

    This tool should be used when the user says things like:
    - "Apply me to Python developer jobs"
    - "Apply for marketing roles in Dhaka using my resume"
    - "Search jobs and apply to the best matching one"

    Args:
        job_query: The job title or keyword to search (e.g., "Data Scientist").
        resume_text: The full text of the user's resume.
        recipient_email: The hiring manager's email address to send the application to (optional).
        location: Optional location to filter jobs (e.g., "Dhaka", "Remote").
    """
    if not resume_text or len(resume_text.strip()) < 50:
        return (
            "⚠️ No resume found. Please upload your resume first by attaching it in the chat "
            "(name the file 'resume.pdf' or 'resume.txt')."
        )

    # Step 1: Search for jobs
    jobs = _search_jobs_internal(job_query, location)
    if not jobs:
        return f"❌ Could not find any jobs for '{job_query}'. Try a different query or check your SerpApi key."

    # Step 2: Pick the first/best job
    target_job = jobs[0]
    title = target_job.get("title", "Unknown Title")
    company = target_job.get("company_name", "Unknown Company")
    loc = target_job.get("location", location or "Not specified")
    description = target_job.get("description", "")[:1500]
    apply_link = target_job.get("share_link", "No link available")

    # Step 3: Generate cover letter
    cover_letter = _generate_cover_letter_internal(
        job_description=f"{title} at {company}\n\n{description}",
        resume_text=resume_text
    )

    # Step 4: Send email (or report the cover letter)
    email_result = ""
    if recipient_email:
        subject = f"Job Application – {title} at {company}"
        email_result = _send_email_internal(
            to=recipient_email,
            subject=subject,
            body=cover_letter
        )
    else:
        email_result = (
            "💡 No recipient email was provided. You can copy the cover letter above and apply directly.\n"
            f"   Apply here: {apply_link}"
        )

    # Build full result summary
    result = (
        f"## 🎯 Job Application Summary\n\n"
        f"**Role:** {title}\n"
        f"**Company:** {company}\n"
        f"**Location:** {loc}\n"
        f"**Apply Link:** {apply_link}\n\n"
        f"---\n\n"
        f"### ✉️ Cover Letter Generated\n\n"
        f"{cover_letter}\n\n"
        f"---\n\n"
        f"### 📬 Delivery Status\n\n"
        f"{email_result}"
    )

    return result
