"""
Resume Optimizer Tool — Modifies the resume to better match a job description.
"""

from langchain_core.tools import tool
from utils.llm import get_llm, get_model_candidates, is_token_limit_error
from langchain_core.messages import HumanMessage, SystemMessage

MAX_JOB_DESC_CHARS = 1500
MAX_RESUME_CHARS = 2500

@tool
def optimize_resume(job_description: str, resume_text: str = "") -> str:
    """
    Optimize and tailor a user's resume to better match a specific job description.
    
    Args:
        job_description: The description or snippet of the job.
        resume_text: The full text of the user's resume.
    """
    if not resume_text:
        return "Error: No resume text provided. Please upload a resume first."

    trimmed_job_description = job_description[:MAX_JOB_DESC_CHARS]
    trimmed_resume_text = resume_text[:MAX_RESUME_CHARS]
        
    system_prompt = (
        "You are an expert resume writer and recruiter. "
        "Your goal is to tailor the provided resume to the job description without fabricating information. "
        "1. Highlight or rephrase existing experiences to better align with the job's keywords. "
        "2. Reorganize skills to put the most relevant ones first. "
        "3. Provide the full, optimized text of the resume, followed by a brief summary of the changes made."
    )
    
    user_prompt = (
        f"--- JOB DESCRIPTION ---\n{trimmed_job_description}\n\n"
        f"--- RESUME ---\n{trimmed_resume_text}\n\n"
        "Please provide the optimized resume."
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
            return f"Error optimizing resume: {str(e)}"

    return f"Error optimizing resume: {str(last_error)}"
