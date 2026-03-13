"""
Job Search Tool — Connects to SerpApi Google Jobs API to find job listings.
"""

import os
from langchain_core.tools import tool
from serpapi import GoogleSearch
from dotenv import load_dotenv

load_dotenv(override=True)

@tool
def search_jobs(query: str, location: str = "") -> str:
    """
    Search for jobs using Google Jobs API (SerpApi).
    
    Args:
        query: The job title or keywords (e.g., 'Python Developer').
        location: The location for the job search (optional).
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        env_path = os.path.abspath(".env")
        exists = os.path.exists(env_path)
        return (
            f"Error: SERPAPI_API_KEY not found in environment.\n"
            f"Looking for .env at: {env_path}\n"
            f".env file exists: {exists}\n"
            "Please ensure the key is correctly set in your .env file."
        )
        
    params = {
      "engine": "google_jobs",
      "q": query,
      "api_key": api_key
    }
    if location:
        params["location"] = location
        
    try:
        search = GoogleSearch(params)
        results = search.get_dict()
        
        if "error" in results:
            return f"Error from SerpApi: {results['error']}"
            
        jobs = results.get("jobs_results", [])
        if not jobs:
            return "No jobs found for the given query."
            
        # Format the top 5 results
        output = f"Top {min(5, len(jobs))} Job Results:\n\n"
        for i, job in enumerate(jobs[:5], 1):
            title = job.get("title", "No Title")
            company = job.get("company_name", "No Company")
            loc = job.get("location", "No Location")
            snippet = job.get("description", "No Description")[:200] + "..."
            apply_link = job.get("share_link", "No Link")
            
            output += f"{i}. {title} at {company}\n"
            output += f"   Location: {loc}\n"
            output += f"   Description: {snippet}\n"
            output += f"   Link: {apply_link}\n\n"
            
        return output
        
    except Exception as e:
        return f"Error searching for jobs: {str(e)}"
