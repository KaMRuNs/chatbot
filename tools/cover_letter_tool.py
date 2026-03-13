"""
Cover Letter Tool — Generates a personalized cover letter.
"""

from langchain_core.tools import tool
from utils.llm import get_llm
from langchain_core.messages import HumanMessage, SystemMessage

@tool
def generate_cover_letter(job_description: str, resume_text: str = "") -> str:
    """
    Generate a personalized cover letter based on the user's resume and the job description.
    
    Args:
        job_description: The description or snippet of the job.
        resume_text: The full text of the user's resume.
    """
    if not resume_text:
        return "Error: No resume text provided. Please upload a resume first."
        
    llm = get_llm()
    
    system_prompt = (
        "You are an expert career coach and copywriter. "
        "Write a compelling, professional, and personalized cover letter. "
        "Use the provided resume to highlight relevant experiences and skills that match the job description. "
        "Keep it concise, engaging, and ready to be sent (leave placeholders like [Company Name] if missing)."
    )
    
    user_prompt = (
        f"--- JOB DESCRIPTION ---\n{job_description}\n\n"
        f"--- RESUME ---\n{resume_text}\n\n"
        "Please draft the cover letter."
    )
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ])
        return response.content
    except Exception as e:
        return f"Error generating cover letter: {str(e)}"
