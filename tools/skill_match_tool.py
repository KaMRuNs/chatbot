"""
Skill Match Tool — Compares user resume with a job description.
"""

from langchain_core.tools import tool
from utils.llm import get_llm, get_model_candidates, is_token_limit_error
from langchain_core.messages import HumanMessage, SystemMessage

MAX_JOB_DESC_CHARS = 1500
MAX_RESUME_CHARS = 2500

@tool
def match_skills(job_description: str, resume_text: str = "") -> str:
    """
    Compare a user's resume against a job description to identify matching skills and missing skills.
    
    Args:
        job_description: The description or snippet of the job.
        resume_text: The full text of the user's resume.
    """
    if not resume_text:
        return "Error: No resume text provided. Please upload a resume first."

    trimmed_job_description = job_description[:MAX_JOB_DESC_CHARS]
    trimmed_resume_text = resume_text[:MAX_RESUME_CHARS]
        
    system_prompt = (
        "You are an expert technical recruiter and career coach. "
        "Analyze the provided resume and job description. "
        "Identify:\n"
        "1. Which skills and experiences from the resume match the job requirements.\n"
        "2. Which critical skills or experiences are missing from the resume.\n"
        "3. Provide specific recommendations for learning the missing skills (e.g., suggest 1-2 free courses, YouTube search terms, or platforms like Coursera/Udemy for each missing skill).\n"
        "Provide a concise, clearly formatted summary."
    )
    
    user_prompt = (
        f"--- JOB DESCRIPTION ---\n{trimmed_job_description}\n\n"
        f"--- RESUME ---\n{trimmed_resume_text}\n\n"
        "Please provide the skill gap analysis."
    )
    
    last_error = None
    for model_name in get_model_candidates():
        llm = get_llm(model_name=model_name)
        try:
            response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])
            return response.content
        except Exception as e:
            last_error = e
            if is_token_limit_error(e):
                continue
            return f"Error matching skills: {str(e)}"

    return f"Error matching skills: {str(last_error)}"
