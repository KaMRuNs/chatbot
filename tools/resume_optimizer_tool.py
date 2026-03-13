"""
Resume Optimizer Tool — Modifies the resume to better match a job description.
"""

from langchain_core.tools import tool
from utils.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

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
        
    llm = get_llm()
    
    system_prompt = (
        "You are an expert resume writer and recruiter. "
        "Your goal is to tailor the provided resume to the job description without fabricating information. "
        "1. Highlight or rephrase existing experiences to better align with the job's keywords. "
        "2. Reorganize skills to put the most relevant ones first. "
        "3. Provide the full, optimized text of the resume, followed by a brief summary of the changes made."
    )
    
    user_prompt = (
        f"--- JOB DESCRIPTION ---\n{job_description}\n\n"
        f"--- RESUME ---\n{resume_text}\n\n"
        "Please provide the optimized resume."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content
    except Exception as e:
        return f"Error optimizing resume: {str(e)}"
