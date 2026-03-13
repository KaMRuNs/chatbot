"""
Skill Match Tool — Compares user resume with a job description.
"""

from langchain_core.tools import tool
from utils.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

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
        
    llm = get_llm()
    
    system_prompt = (
        "You are an expert technical recruiter and career coach. "
        "Analyze the provided resume and job description. "
        "Identify:\n"
        "1. Which skills and experiences from the resume match the job requirements.\n"
        "2. Which critical skills or experiences are missing from the resume.\n"
        "Provide a concise, clearly formatted summary."
    )
    
    user_prompt = (
        f"--- JOB DESCRIPTION ---\n{job_description}\n\n"
        f"--- RESUME ---\n{resume_text}\n\n"
        "Please provide the skill gap analysis."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content
    except Exception as e:
        return f"Error matching skills: {str(e)}"
